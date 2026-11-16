import { compile } from './pattern.mjs';

self.onmessage = ({ data }) => {
  try {
    self.postMessage({ score: compile(data) });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
