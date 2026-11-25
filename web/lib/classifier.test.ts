// web/lib/classifier.test.ts
import { describe, expect, it } from "vitest";
import { predict, predictProba, type ForestModel } from "./classifier";

function makeStumpModel(): ForestModel {
  // A single tree with one split on feature 0 at threshold 0.0: left leaf favors
  // class 0 (10 votes), right leaf favors class 1 (10 votes). childrenLeft/Right
  // use -1 for leaves, feature/threshold use -2 for leaves (matches sklearn's
  // tree_ convention).
  const tree = {
    feature: [0, -2, -2],
    threshold: [0.0, -2.0, -2.0],
    childrenLeft: [1, -1, -1],
    childrenRight: [2, -1, -1],
    value: [[0, 0], [10, 0], [0, 10]],
  };
  return {
    classes: ["CorrectSitting", "LeanForward"],
    featureOrder: ["x"],
    scaleReference: 1.0,
    trees: [tree],
  };
}

describe("classifier tree-walk", () => {
  it("predicts the left leaf's class when the feature is below threshold", () => {
    const result = predict(makeStumpModel(), [-1.0]);
    expect(result.label).toBe("CorrectSitting");
    expect(result.confidence).toBeCloseTo(100.0, 1);
  });

  it("predicts the right leaf's class when the feature is above threshold", () => {
    const result = predict(makeStumpModel(), [1.0]);
    expect(result.label).toBe("LeanForward");
  });

  it("averages per-class probabilities across multiple trees", () => {
    const model = makeStumpModel();
    model.trees.push(model.trees[0]);
    const proba = predictProba(model, [-1.0]);
    expect(proba[0]).toBeCloseTo(1.0, 5);
    expect(proba[1]).toBeCloseTo(0.0, 5);
  });
});
