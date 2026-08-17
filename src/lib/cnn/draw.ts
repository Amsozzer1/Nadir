import matrix from "./matrix";

export const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";
export const DIM = "#7a7a7a";
export const HI = "#ffffff";

export function peakOf(m: matrix): number {
    let p = 0;
    for (let i = 0; i < m.rows; i++) {
        for (let j = 0; j < m.cols; j++) {
            const a = Math.abs(m.values[i][j]);
            if (a > p) p = a;
        }
    }
    return p || 1;
}

// Positives run white, negatives run blue. Encoding the sign as hue rather
// than as darkness is what makes ReLU visible -- otherwise "negative" and
// "zero" both render as black and the step looks like nothing happened.
export function shade(v: number, peak: number): string {
    const u = Math.min(1, Math.abs(v) / peak);
    const c = Math.round(u * 255);
    return v < 0
        ? `rgb(${(c * 0.35) | 0},${(c * 0.6) | 0},${c})`
        : `rgb(${c},${c},${c})`;
}

export type MapOpts = {
    peak?: number;
    upto?: number;
    relu?: boolean;
    alpha?: number;
};

export function drawMatrix(
    ctx: CanvasRenderingContext2D,
    m: matrix,
    x: number,
    y: number,
    cell: number,
    o: MapOpts = {},
) {
    const peak = o.peak ?? peakOf(m);
    const total = m.rows * m.cols;
    const n = Math.min(o.upto ?? total, total);
    ctx.save();
    if (o.alpha !== undefined) ctx.globalAlpha = o.alpha;
    ctx.fillStyle = "#000";
    ctx.fillRect(x, y, m.cols * cell, m.rows * cell);
    for (let k = 0; k < n; k++) {
        const i = (k / m.cols) | 0;
        const j = k % m.cols;
        let v = m.values[i][j];
        if (o.relu && v < 0) v = 0;
        if (v === 0) continue;
        ctx.fillStyle = shade(v, peak);
        ctx.fillRect(x + j * cell, y + i * cell, cell, cell);
    }
    ctx.restore();
}

export function outline(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    color = HI,
    lw = 1,
) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    ctx.restore();
}

export function label(
    ctx: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    color = DIM,
    size = 11,
    align: CanvasTextAlign = "left",
) {
    ctx.save();
    ctx.font = `${size}px ${MONO}`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = "top";
    ctx.fillText(text, x, y);
    ctx.restore();
}

export function line(
    ctx: CanvasRenderingContext2D,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color = DIM,
    alpha = 1,
) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
}

export function arrowDown(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    len: number,
    color = DIM,
) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + len - 5);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y + len);
    ctx.lineTo(x - 3.5, y + len - 6);
    ctx.lineTo(x + 3.5, y + len - 6);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
}

export function arrow(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    len: number,
    color = DIM,
) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + len - 5, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + len, y);
    ctx.lineTo(x + len - 6, y - 3.5);
    ctx.lineTo(x + len - 6, y + 3.5);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
}
