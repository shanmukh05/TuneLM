# Source research — 2026-11-06

The user authorized additional Strudel-licensed collections and requested no raw source-file retention. The existing allowlist was extended from 2 to 19 scopes. All prior candidates, accepted seeds, and stems were preserved. Downloads now live only in a temporary directory; extracted JSONL, source URLs/checksums, and license notices remain. No teacher APIs or training ran.

Collection found 699 candidates before exact code deduplication: 679 distinct candidates, 20 duplicates, **516 accepted compositions**, **662 verified stems**, and 163 rejections. This adds 413 compositions and 477 stems to the initial snapshot. Native verification establishes execution/catalog compatibility, not listening quality.

## Imported scopes

Counts in the extracted column precede exact deduplication. Zero accepted entries means the source remains available as candidate data with saved rejection diagnostics; it is not part of the accepted seed bank. Every file, revision, checksum, and license scope is pinned in [`configs/registry/sources.yaml`](../configs/registry/sources.yaml).

| Collection | License | Extracted | Accepted | Stems |
| --- | --- | ---: | ---: | ---: |
| [arnoldwender/wm-prototyp-live-music-coder](https://github.com/arnoldwender/wm-prototyp-live-music-coder) | MIT | 130 | 93 | 175 |
| [thedavidweng/strudel-tui](https://github.com/thedavidweng/strudel-tui) | MIT | 6 | 0 | 0 |
| [sueszli/strudel-recipes](https://github.com/sueszli/strudel-recipes) | AGPL-3.0-only | 2 | 1 | 1 |
| [sthefanygsa/STRUDEL-PATCHES](https://github.com/sthefanygsa/STRUDEL-PATCHES) | MIT | 7 | 1 | 4 |
| [BasClason/StrudelScripts](https://github.com/BasClason/StrudelScripts) | CC0-1.0 | 3 | 0 | 0 |
| [terefang/strudel-tunes](https://github.com/terefang/strudel-tunes) | CC0-1.0 | 9 | 0 | 0 |
| [atraeus7/strudel_loops](https://github.com/atraeus7/strudel_loops) | CC0-1.0 | 3 | 0 | 0 |
| [JensRantil/strudel-collab](https://github.com/JensRantil/strudel-collab) | CC0-1.0 | 1 | 0 | 0 |
| [stormouse/code-the-dj](https://github.com/stormouse/code-the-dj) | CC0-1.0 | 1 | 0 | 0 |
| [Aphexina/Strudel](https://github.com/Aphexina/Strudel) | MIT | 6 | 6 | 17 |
| [Raiaaa17/strudel-code-collection](https://github.com/Raiaaa17/strudel-code-collection) | MIT | 2 | 1 | 1 |
| [anhussain90/strudel-compositions](https://github.com/anhussain90/strudel-compositions) | MIT | 1 | 0 | 0 |
| [karmaterminal/strudel-music](https://github.com/karmaterminal/strudel-music) | MIT | 14 | 9 | 48 |
| [abhirajdg/strudel](https://github.com/abhirajdg/strudel) | CC0-1.0 | 3 | 0 | 0 |
| [cormac85/strudeling](https://github.com/cormac85/strudeling) | CC0-1.0 | 1 | 0 | 0 |
| [AustinOliverHaskell/strudel_code](https://github.com/AustinOliverHaskell/strudel_code) | CC0-1.0 | 1 | 0 | 0 |
| [neverclear86/strudel-records](https://github.com/neverclear86/strudel-records) | CC0-1.0 | 3 | 0 | 0 |
| [williamzujkowski/live-coding-music-mcp](https://github.com/williamzujkowski/live-coding-music-mcp) | AGPL-3.0-or-later | 4 | 4 | 15 |
| [uzu/strudel](https://codeberg.org/uzu/strudel) | GPL-3.0-only | 492 | 391 | 391 |

The ten local starter compositions are additional to the table.

## Important source distinctions

- **Strudel rhythm library:** only the 492 exported program strings in `website/src/repl/drum_patterns.mjs` were selected. Each credits `lvm/tidal-drum-patterns` under GPL-3.0 and the `urswilke/read_beats` converter. Those comments remain in the dataset code; the original GPL text is retained. The surrounding application license was not substituted for the file-level notice. After deduplication and native verification, 391 patterns passed. They share the upstream rhythm-family identity to keep variants together.
- **Live Coding Music MCP:** its `SOURCES.md` distinguishes four original hand-written examples (AGPL-3.0-or-later, as declared upstream) from three GPL-derived rhythm examples. Only the four originals were added; the GPL-derived ones would overlap the original rhythm bank.
- **Independent patches:** original repository/file notices were reviewed for the selected scopes. Methods of authorship are unverified unless the repository declares them; the Dandelion Cult collection describes an agent-oriented workflow. Its audio-derived reconstructions and parameterized template were excluded.
- **Codebooks:** independent top-level pattern expressions retain their preceding setup. Named channel labels may be canonicalized to `$:` while underscore-muted channels stay omitted. Other unsupported programs are left for the verifier to reject. Markdown without a language tag and JSON `code`/`pattern` records are now supported. No downloaded module is imported or executed.

## Reviewed but not imported

| Source/scope | Reason |
| --- | --- |
| DYenn00/Strudel Wasteland files | File-level music comments specify CC BY-NC-SA despite the root MIT notice; the root license was not substituted. |
| rcasamajo/Taller_Strudel | A CC0 workshop guide linking elsewhere, with no inline program fences to collect. |
| grokbeer/strudel | Programs explicitly refer to external MIDI; composition provenance was not resolved in this pass. |
| BasClason Pannekoek; sueszli trance.js | Explicitly reference borrowed external code; the reviewed subset omits them. |
| atraeus7 external remixes/game-cover examples | Only the selected author-credited original scripts were added. |
| aeron-glasgow-keene/strudel | Explicitly describes its track as an adjusted copy of external code. |
| xyflow/strudel-flow | A visual graph preset system; importing graph-state objects as executable code needs an application-specific adapter. |
| nythingwrks/strudel; tofu89/strudel | No composition source files in the inspected trees. |
| Other flagship/community Strudel tunes | Separate composition-level terms; not included by assuming the application license. |

## Remaining constraints

The corpus is rhythm-heavy: 453 stems are classified drums_core, compared with 60 bass and 70 chords_pad. All newly accepted families currently map to training; validation and benchmark still have only two starter compositions each. More independent harmonic/arrangement sources and held-out material are needed before production evaluation. The 1,500-stem target remains ahead.

Unknown sounds, browser controls, custom loaders/extensions, and unsupported syntax remain rejected. No verifier gates were weakened to increase the counts. Evidence and per-source metrics are saved in `results/verification/source-expansion.json`; failures are in `results/ingestion/rejected.jsonl`.

Offline ingestion reuses verified parsed snapshots with no raw files. Configuration or input/hash changes require refreshing the collection online. The legacy archive is unchanged.
