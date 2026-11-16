"""Bridge the loopback-only Studio preview to the existing Modal client profile."""

import json
import os
import sys


def legacy_prompt(prompt, history):
    """Fit recent revisions in the prompt for SDK deployments with only predict()."""
    from tunelm.composition.responses import strudel_excerpt

    turns = []
    remaining = 5400 - len(prompt)
    for turn in reversed(history):
        request = turn["prompt"]
        limit = remaining - len(request) - 100
        if limit < 350 or (turns and len(turn["code"]) > limit):
            break
        code = strudel_excerpt(turn["code"], limit)
        context = f"Earlier listener request: {request}\nEarlier Strudel:\n```strudel\n{code}\n```"
        turns.insert(0, context)
        remaining -= len(context)
    turns.append(
        f"Current listener request: {prompt}\n"
        "For a revision, keep details the listener did not ask to change. Return the full program."
    )
    return "\n\n".join(turns)


def start_job(composer, prompt, history, missing_method):
    if history:
        try:
            return composer.converse.spawn(prompt, history)
        except missing_method:
            prompt = legacy_prompt(prompt, history)
    return composer.predict.spawn(prompt)


def request(method, body):
    try:
        import modal
    except ImportError:
        return 503, {"error": "Install the Modal extra with uv sync --extra modal."}

    app = os.environ.get("TUNELM_MODAL_APP", "tunelm-sft-inference")
    try:
        if method == "POST":
            composer = modal.Cls.from_name(app, "Composer")()
            call = start_job(
                composer, body["prompt"], body.get("history", []), modal.exception.NotFoundError
            )
            return 202, {"job_id": call.object_id}
        return 200, modal.FunctionCall.from_id(body["job"]).get(timeout=0)
    except modal.exception.AuthError:
        return 503, {"error": "Run modal setup to connect this preview to your Modal workspace."}
    except (modal.exception.NotFoundError, modal.exception.OutputExpiredError):
        if method == "POST":
            return 503, {"error": f"The local Modal profile could not find Composer in {app}."}
        return 404, {}
    except modal.exception.FunctionTimeoutError:
        return 504, {}
    except (TimeoutError, modal.exception.TimeoutError):
        if method == "GET":
            return 202, {"status": "pending"}
        raise


if __name__ == "__main__":
    status, body = request(sys.argv[1], json.load(sys.stdin))
    print(json.dumps({"status": status, "body": body}))
