/* ─── Dataset Generation ─── */
function seededRandom(seed) {
    let s = seed;
    return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}
export function generateSVMData(dataset, count, noise) {
    const rand = seededRandom(count * 1000 + Math.floor(noise * 100));
    const pts = [];
    const half = Math.floor(count / 2);
    for (let i = 0; i < count; i++) {
        const cls = i < half ? 0 : 1;
        let x = 0, y = 0;
        if (dataset === 'blobs') {
            const cx = cls === 0 ? 0.3 : 0.7;
            const cy = cls === 0 ? 0.3 : 0.7;
            x = cx + (rand() - 0.5) * noise * 1.5;
            y = cy + (rand() - 0.5) * noise * 1.5;
        }
        else if (dataset === 'linear') {
            x = rand();
            y = rand();
            if (y > x + (rand() - 0.5) * noise) {
                pts.push({ x, y, cls: 0 });
            }
            else {
                pts.push({ x, y, cls: 1 });
            }
            continue;
        }
        else if (dataset === 'moons') {
            const angle = rand() * Math.PI;
            if (cls === 0) {
                x = 0.5 + Math.cos(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
                y = 0.6 - Math.sin(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
            }
            else {
                x = 0.5 - Math.cos(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
                y = 0.4 + Math.sin(angle) * 0.25 + (rand() - 0.5) * noise * 0.5;
            }
        }
        else if (dataset === 'circles') {
            const radius = cls === 0 ? 0.15 : 0.35;
            const angle = rand() * Math.PI * 2;
            x = 0.5 + Math.cos(angle) * radius + (rand() - 0.5) * noise * 0.5;
            y = 0.5 + Math.sin(angle) * radius + (rand() - 0.5) * noise * 0.5;
        }
        else if (dataset === 'xor') {
            x = rand();
            y = rand();
            const nx = x + (rand() - 0.5) * noise * 0.5;
            const ny = y + (rand() - 0.5) * noise * 0.5;
            const isCls1 = (nx > 0.5 && ny > 0.5) || (nx < 0.5 && ny < 0.5);
            pts.push({ x: Math.max(0, Math.min(1, nx)), y: Math.max(0, Math.min(1, ny)), cls: isCls1 ? 1 : 0 });
            continue;
        }
        else if (dataset === 'spiral') {
            const n = count / 2;
            const r = (i % n) / n * 0.4;
            const t = 1.25 * (i % n) / n * 2 * Math.PI + (cls === 1 ? Math.PI : 0);
            x = 0.5 + r * Math.sin(t) + (rand() - 0.5) * noise * 0.1;
            y = 0.5 + r * Math.cos(t) + (rand() - 0.5) * noise * 0.1;
        }
        pts.push({ x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)), cls });
    }
    // Linear needs truncation because of the random rejection logic
    return pts.slice(0, count);
}
/* ─── SVM Math ─── */
// Features mapping for non-linear boundaries via polynomial kernel equivalent
export function expandFeatures(x1, x2, degree) {
    if (degree === 1)
        return [x1, x2];
    if (degree === 2)
        return [x1, x2, x1 * x1, x2 * x2, x1 * x2];
    return [x1, x2];
}
/** Get the number of features for a given degree */
export function featureCount(degree) {
    if (degree === 2)
        return 5;
    return 2;
}
/** Initialize weights with random values (breaks symmetry) */
export function initWeights(degree, seed = 42) {
    const n = featureCount(degree) + 1; // +1 for bias
    const rand = seededRandom(seed);
    return Array.from({ length: n }, () => (rand() - 0.5) * 2.0);
}
// Compute f(x) = w^T x + b
export function computeMargin(px, py, weights, degree) {
    if (weights.length === 0)
        return 0;
    const b = weights[0];
    const w = weights.slice(1);
    const feats = expandFeatures(px, py, degree);
    let dot = b;
    for (let i = 0; i < feats.length; i++) {
        dot += w[i] * feats[i];
    }
    return dot;
}
export function predict(px, py, weights, degree) {
    const marginDist = computeMargin(px, py, weights, degree);
    // Prob is mapped through a sigmoid just for visualizing confidence, even though SVM doesn't output true probs
    const prob = 1 / (1 + Math.exp(-marginDist));
    const cls = marginDist >= 0 ? 1 : 0;
    return { cls, prob, marginDist };
}
// Hinge Loss: L = C * sum(max(0, 1 - y_i * f(x_i))) + 0.5 * ||w||^2
// y_i in {-1, 1}
// This is the standard formulation (like scikit-learn) where C multiplies the sum.
export function computeLoss(points, weights, degree, C) {
    if (points.length === 0 || weights.length === 0)
        return 0;
    let hingeSum = 0;
    const w = weights.slice(1);
    for (const pt of points) {
        const yTrue = pt.cls === 1 ? 1 : -1;
        const fX = computeMargin(pt.x, pt.y, weights, degree);
        hingeSum += Math.max(0, 1 - yTrue * fX);
    }
    // L2 Regularization term
    let l2 = 0;
    for (const val of w)
        l2 += val * val;
    return C * hingeSum + 0.5 * l2;
}
export function computeGradients(points, weights, degree, C) {
    if (points.length === 0 || weights.length === 0)
        return weights.map(() => 0);
    const grads = new Array(weights.length).fill(0);
    const w = weights.slice(1);
    // Regularization gradient: d/dw (0.5 * ||w||^2) = w (no regularization on bias)
    for (let i = 0; i < w.length; i++) {
        grads[i + 1] = w[i];
    }
    // Hinge loss gradient: d/dw (C * sum max(0, 1 - y*f(x)))
    for (const pt of points) {
        const yTrue = pt.cls === 1 ? 1 : -1;
        const fX = computeMargin(pt.x, pt.y, weights, degree);
        if (1 - yTrue * fX > 0) {
            grads[0] -= C * yTrue; // bias gradient
            const feats = expandFeatures(pt.x, pt.y, degree);
            for (let j = 0; j < feats.length; j++) {
                grads[j + 1] -= C * yTrue * feats[j];
            }
        }
    }
    // Clip gradients to prevent exploding gradients
    let normSq = 0;
    for (const g of grads)
        normSq += g * g;
    const norm = Math.sqrt(normSq);
    const maxNorm = 100.0;
    if (norm > maxNorm) {
        const scale = maxNorm / norm;
        for (let i = 0; i < grads.length; i++)
            grads[i] *= scale;
    }
    return grads;
}
/** Run a single PEGASOS-style SGD step (mutates and returns weights) */
export function trainStep(weights, points, degree, C, lr, step) {
    // Gentle learning rate decay
    const effectiveLr = lr / (1 + step * 0.0005);
    const grads = computeGradients(points, weights, degree, C);
    const newW = weights.map((w, i) => w - effectiveLr * grads[i]);
    const loss = computeLoss(points, newW, degree, C);
    return { weights: newW, loss };
}
export function computeConfusionMatrix(points, weights, degree) {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const pt of points) {
        const { cls } = predict(pt.x, pt.y, weights, degree);
        if (cls === 1 && pt.cls === 1)
            tp++;
        if (cls === 0 && pt.cls === 0)
            tn++;
        if (cls === 1 && pt.cls === 0)
            fp++;
        if (cls === 0 && pt.cls === 1)
            fn++;
    }
    return { tp, tn, fp, fn };
}
export function computeMetrics(points, weights, degree) {
    if (points.length === 0 || weights.length === 0)
        return [
            { label: 'Accuracy', value: '—', isPrimary: true },
            { label: 'Precision', value: '—' },
            { label: 'Recall', value: '—' },
            { label: 'F1 Score', value: '—' },
        ];
    const { tp, tn, fp, fn } = computeConfusionMatrix(points, weights, degree);
    const accuracy = (tp + tn) / points.length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    return [
        { label: 'Accuracy', value: `${(accuracy * 100).toFixed(1)}%`, isPrimary: true },
        { label: 'Precision', value: `${(precision * 100).toFixed(1)}%` },
        { label: 'Recall', value: `${(recall * 100).toFixed(1)}%` },
        { label: 'F1 Score', value: `${(f1 * 100).toFixed(1)}%` },
    ];
}
/** Count support vectors — points on or inside the margin */
export function countSupportVectors(points, weights, degree) {
    let count = 0;
    for (const pt of points) {
        const yTrue = pt.cls === 1 ? 1 : -1;
        const m = computeMargin(pt.x, pt.y, weights, degree);
        if (1 - yTrue * m >= 0)
            count++;
    }
    return count;
}
export function computeDataStats(points, weights, degree) {
    if (points.length === 0)
        return null;
    const n = points.length;
    const nClass0 = points.filter(p => p.cls === 0).length;
    const nClass1 = n - nClass0;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of points) {
        if (p.x < xMin)
            xMin = p.x;
        if (p.x > xMax)
            xMax = p.x;
        if (p.y < yMin)
            yMin = p.y;
        if (p.y > yMax)
            yMax = p.y;
    }
    const supportVectors = countSupportVectors(points, weights, degree);
    return { n, nClass0, nClass1, xRange: [xMin, xMax], yRange: [yMin, yMax], supportVectors };
}
export function formatEquation(weights, degree) {
    if (weights.length === 0)
        return 'f(x) = ...';
    const b = weights[0].toFixed(2);
    const w = weights.slice(1).map(v => v.toFixed(2));
    let eq = `f(x) = ${w[0]}x₁ `;
    eq += parseFloat(w[1]) >= 0 ? `+ ${w[1]}x₂ ` : `- ${Math.abs(parseFloat(w[1]))}x₂ `;
    if (degree === 2) {
        eq += parseFloat(w[2]) >= 0 ? `+ ${w[2]}x₁² ` : `- ${Math.abs(parseFloat(w[2]))}x₁² `;
        eq += parseFloat(w[3]) >= 0 ? `+ ${w[3]}x₂² ` : `- ${Math.abs(parseFloat(w[3]))}x₂² `;
        eq += parseFloat(w[4]) >= 0 ? `+ ${w[4]}x₁x₂ ` : `- ${Math.abs(parseFloat(w[4]))}x₁x₂ `;
    }
    eq += parseFloat(b) >= 0 ? `+ ${b}` : `- ${Math.abs(parseFloat(b))}`;
    return eq;
}
