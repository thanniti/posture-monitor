// web/lib/classifier.ts
export interface ForestTree {
  feature: number[];
  threshold: number[];
  childrenLeft: number[];
  childrenRight: number[];
  value: number[][];
}

export interface ForestModel {
  classes: string[];
  featureOrder: string[];
  scaleReference: number;
  trees: ForestTree[];
}

function predictTreeProba(tree: ForestTree, features: number[]): number[] {
  let node = 0;
  while (tree.childrenLeft[node] !== -1) {
    const goLeft = features[tree.feature[node]] <= tree.threshold[node];
    node = goLeft ? tree.childrenLeft[node] : tree.childrenRight[node];
  }
  const counts = tree.value[node];
  const total = counts.reduce((a, b) => a + b, 0);
  return counts.map((c) => c / total);
}

export function predictProba(model: ForestModel, features: number[]): number[] {
  const nClasses = model.classes.length;
  const sums = new Array(nClasses).fill(0);
  for (const tree of model.trees) {
    const proba = predictTreeProba(tree, features);
    for (let i = 0; i < nClasses; i++) sums[i] += proba[i];
  }
  return sums.map((s) => s / model.trees.length);
}

export function predict(model: ForestModel, features: number[]): { label: string; confidence: number } {
  const proba = predictProba(model, features);
  let bestIndex = 0;
  for (let i = 1; i < proba.length; i++) {
    if (proba[i] > proba[bestIndex]) bestIndex = i;
  }
  return { label: model.classes[bestIndex], confidence: Math.round(proba[bestIndex] * 1000) / 10 };
}

export async function loadModel(url: string): Promise<ForestModel> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load classifier model from ${url}: ${response.status}`);
  }
  return response.json();
}
