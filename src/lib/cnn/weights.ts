import matrix from "./matrix";
import { Checkpoint, Layer, Model } from "./types";

export const WEIGHTS_URL = "/epoch3.json";

const cache = new Map<string, Promise<Model>>();

export function loadWeights(url: string = WEIGHTS_URL): Promise<Model> {
    let pending = cache.get(url);
    if (!pending) {
        pending = fetchCheckpoint(url).catch((e) => {
            cache.delete(url);
            throw e;
        });
        cache.set(url, pending);
    }
    return pending;
}

async function fetchCheckpoint(url: string): Promise<Model> {
    const res = await fetch(url);
    if (!res.ok) throw Error(`Weights fetch failed: ${res.status} ${res.statusText} (${url})`);
    return parseCheckpoint(await res.json(), url);
}

export function parseCheckpoint(d: Checkpoint, src: string = "checkpoint"): Model {
    if (!d || !Array.isArray(d.weights) || !Array.isArray(d.biases) || !Array.isArray(d.kernels)) {
        throw Error(`${src}: missing weights/biases/kernels`);
    }
    if (d.weights.length !== 2 || d.biases.length !== 2) {
        throw Error(`${src}: expected 2 dense layers, got ${d.weights.length} weights / ${d.biases.length} biases`);
    }

    const dense = d.weights.map((v, i) => grid(v, `${src}: weights[${i}]`));

    // Mirrors the assertion in the trainer's load_checkpoint.
    if (Array.isArray(d.shapes)) {
        for (let i = 0; i < dense.length; i++) {
            const [r, c] = d.shapes[i];
            if (dense[i].rows !== r || dense[i].cols !== c) {
                throw Error(`${src}: weights[${i}] is ${dense[i].rows}x${dense[i].cols}, shapes says ${r}x${c}`);
            }
        }
    }
    for (let i = 0; i < dense.length; i++) {
        if (d.biases[i].length !== dense[i].rows) {
            throw Error(`${src}: biases[${i}] has ${d.biases[i].length} entries, weights[${i}] has ${dense[i].rows} rows`);
        }
    }
    if (dense[1].cols !== dense[0].rows) {
        throw Error(`${src}: W2 takes ${dense[1].cols} inputs but W1 emits ${dense[0].rows}`);
    }

    const layers: Array<Layer> = d.kernels.map((filters, i) => {
        if (!Array.isArray(filters) || filters.length === 0) {
            throw Error(`${src}: kernels[${i}] has no filters`);
        }
        const stacks = filters.map((stack, j) => {
            if (!Array.isArray(stack) || stack.length === 0) {
                throw Error(`${src}: kernels[${i}][${j}] has no channels`);
            }
            return stack.map((k, c) => {
                const m = grid(k, `${src}: kernels[${i}][${j}][${c}]`);
                if (m.rows !== m.cols) {
                    throw Error(`${src}: kernels[${i}][${j}][${c}] is ${m.rows}x${m.cols}, conv2D needs a square kernel`);
                }
                return m;
            });
        });
        return { filters: stacks, stride: 1 };
    });

    for (let i = 1; i < layers.length; i++) {
        const c_in = layers[i].filters[0].length;
        const c_out = layers[i - 1].filters.length;
        if (c_in !== c_out) {
            throw Error(`${src}: layer ${i} expects ${c_in} channels, layer ${i - 1} emits ${c_out}`);
        }
    }

    return {
        epoch: d.epoch,
        layers,
        W1: dense[0],
        b1: d.biases[0],
        W2: dense[1],
        b2: d.biases[1],
    };
}

function grid(v: Array<Array<number>>, what: string): matrix {
    if (!Array.isArray(v) || v.length === 0 || !Array.isArray(v[0]) || v[0].length === 0) {
        throw Error(`${what}: expected a non-empty 2D array`);
    }
    for (let i = 0; i < v.length; i++) {
        if (v[i].length !== v[0].length) {
            throw Error(`${what}: row ${i} has ${v[i].length} entries, row 0 has ${v[0].length}`);
        }
    }
    return new matrix(v);
}
