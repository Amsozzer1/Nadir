import matrix from "./matrix";

// Feature maps are post-ReLU, so the floor is already 0. Each map is scaled by
// its own peak rather than a shared one -- filters fire at wildly different
// magnitudes, and a global scale renders the quiet ones as flat black.
export function featureMapImage(m: matrix): ImageData {
    const img = new ImageData(m.cols, m.rows);
    const peak = m._max();
    const k = peak > 0 ? 255 / peak : 0;
    for (let y = 0; y < m.rows; y++) {
        for (let x = 0; x < m.cols; x++) {
            const v = m.values[y][x] * k;
            const o = (y * m.cols + x) * 4;
            img.data[o] = v;
            img.data[o + 1] = v;
            img.data[o + 2] = v;
            img.data[o + 3] = 255;
        }
    }
    return img;
}
