import matrix from "./matrix";

const SIZE = 28;
const BOX = 20;

// Canvas -> 28x28, matching how MNIST was built: the digit is scaled to fit a
// 20x20 box preserving aspect, then translated so its centre of mass sits at
// the middle of the field. Ink is read from the alpha channel, so the source
// canvas must be transparent with opaque strokes -- that gives 0.0 background
// and 1.0 ink directly, which is the polarity the trainer used.
export function toInput(source: HTMLCanvasElement): matrix | null {
    const sctx = source.getContext("2d");
    if (!sctx) return null;

    const w = source.width;
    const h = source.height;
    const px = sctx.getImageData(0, 0, w, h).data;

    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (px[(y * w + x) * 4 + 3] === 0) continue;
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            if (y < y0) y0 = y;
            if (y > y1) y1 = y;
        }
    }
    if (x1 < 0) return null;

    const bw = x1 - x0 + 1;
    const bh = y1 - y0 + 1;
    const scale = BOX / Math.max(bw, bh);
    const sw = Math.max(1, Math.round(bw * scale));
    const sh = Math.max(1, Math.round(bh * scale));

    const tmp = document.createElement("canvas");
    tmp.width = SIZE;
    tmp.height = SIZE;
    const tctx = tmp.getContext("2d");
    if (!tctx) return null;
    tctx.drawImage(source, x0, y0, bw, bh, (SIZE - sw) / 2, (SIZE - sh) / 2, sw, sh);
    const small = tctx.getImageData(0, 0, SIZE, SIZE).data;

    const grid: number[][] = Array.from({ length: SIZE }, () => Array(SIZE).fill(0.0));
    let mass = 0, mx = 0, my = 0;
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const v = small[(y * SIZE + x) * 4 + 3] / 255;
            grid[y][x] = v;
            mass += v;
            mx += x * v;
            my += y * v;
        }
    }
    if (mass === 0) return null;

    const dx = Math.round(SIZE / 2 - mx / mass);
    const dy = Math.round(SIZE / 2 - my / mass);
    if (dx === 0 && dy === 0) return new matrix(grid);

    const out: number[][] = Array.from({ length: SIZE }, () => Array(SIZE).fill(0.0));
    for (let y = 0; y < SIZE; y++) {
        const sy = y - dy;
        if (sy < 0 || sy >= SIZE) continue;
        for (let x = 0; x < SIZE; x++) {
            const sx = x - dx;
            if (sx < 0 || sx >= SIZE) continue;
            out[y][x] = grid[sy][sx];
        }
    }
    return new matrix(out);
}
