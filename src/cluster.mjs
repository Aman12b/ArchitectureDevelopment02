// Embeds the functional EarlyBird requirements, stores them in a DuckDB vector
// database and lets the database cluster them: k-means (k-means++ init,
// assignment, centroid update) and the silhouette score run as SQL inside
// DuckDB; JavaScript only orchestrates the iterations. Usage: npm start
import fs from "node:fs";
import path from "node:path";
import { pipeline } from "@huggingface/transformers";
import { DuckDBInstance } from "@duckdb/node-api";

const ROOT = path.resolve(import.meta.dirname, "..");
const OUT = path.join(ROOT, "output");
const DB_FILE = path.join(ROOT, "vectordb", "earlybird.duckdb");
const MODEL = "Xenova/all-MiniLM-L6-v2";
const DIM = 384;
const K_RANGE = [4, 8]; // course guideline: 4-6 building blocks; silhouette is nearly flat (see README)
const RESTARTS = 30;
const MAX_ITER = 100;
const SEED = 0.42;

const requirements = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "requirements.json"), "utf8"));
const namesFile = path.join(ROOT, "data", "cluster-names.json");
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(path.dirname(DB_FILE), { recursive: true });
fs.rmSync(DB_FILE, { force: true });

// 1. Embeddings (normalized, so cosine distance = 1 - dot product)
console.log(`Embedding ${requirements.length} requirements with ${MODEL} ...`);
const embed = await pipeline("feature-extraction", MODEL, { dtype: "fp32" });
const tensor = await embed(requirements.map((r) => r.text), { pooling: "mean", normalize: true });
const embeddings = tensor.tolist();

// 2. Vector database
const db = await (await DuckDBInstance.create(DB_FILE, { threads: "1" })).connect(); // 1 thread => reproducible random()
const run = (sql) => db.run(sql);
const all = async (sql) => (await db.runAndReadAll(sql)).getRowObjectsJson();
const one = async (sql) => Object.values((await all(sql))[0])[0];
const sqlString = (s) => `'${s.replaceAll("'", "''")}'`;

await run(`CREATE TABLE requirements (id INTEGER PRIMARY KEY, text VARCHAR, embedding FLOAT[${DIM}])`);
await run(`INSERT INTO requirements VALUES ${requirements
  .map((r, i) => `(${r.id}, ${sqlString(r.text)}, [${embeddings[i].join(",")}]::FLOAT[${DIM}])`)
  .join(",\n")}`);
console.log(`Stored ${await one("SELECT count(*) FROM requirements")} vectors in DuckDB (${path.relative(ROOT, DB_FILE)}).`);

// 3. Clustering inside the database
await run(`CREATE TEMP TABLE centroids (cluster INTEGER, centroid FLOAT[${DIM}])`);
await run(`CREATE TEMP TABLE assignments (id INTEGER, cluster INTEGER)`);
await run(`CREATE TABLE kmeans_runs (k INTEGER, restart INTEGER, iterations INTEGER, inertia DOUBLE, silhouette DOUBLE)`);
await run(`CREATE TABLE assignments_by_k (k INTEGER, id INTEGER, cluster INTEGER)`);

// k-means++: first centroid uniformly at random, every further one with probability ~ D^2
// (weighted sampling via ORDER BY random()^(1/w) DESC, Efraimidis-Spirakis)
// DuckDB derives random() per query from the setseed value, so every query gets its own seed
const seed = (restart, c) => run(`SELECT setseed(${(SEED + (restart * 16 + c) / 1000).toFixed(4)})`);

async function initCentroids(k, restart) {
  await run(`DELETE FROM centroids`);
  await seed(restart, 0);
  await run(`INSERT INTO centroids SELECT 0, embedding FROM requirements ORDER BY random() LIMIT 1`);
  for (let c = 1; c < k; c++) {
    await seed(restart, c);
    await run(`
      INSERT INTO centroids
      SELECT ${c}, r.embedding
      FROM requirements r, centroids ct
      GROUP BY r.id, r.embedding
      HAVING min(array_cosine_distance(r.embedding, ct.centroid)) > 1e-9
      ORDER BY pow(random(), 1.0 / pow(min(array_cosine_distance(r.embedding, ct.centroid)), 2)) DESC
      LIMIT 1`);
  }
}

// Assignment step: every requirement goes to its nearest centroid
const ASSIGN = `
  SELECT r.id, arg_min(ct.cluster, array_cosine_distance(r.embedding, ct.centroid)) AS cluster
  FROM requirements r, centroids ct GROUP BY r.id`;

// Update step: centroid = element-wise mean of the member vectors
const UPDATE_CENTROIDS = `
  INSERT INTO centroids
  SELECT cluster, list(v ORDER BY d)::FLOAT[${DIM}]
  FROM (
    SELECT a.cluster, d, avg(r.embedding[d]) AS v
    FROM assignments a JOIN requirements r USING (id), range(1, ${DIM + 1}) t(d)
    GROUP BY a.cluster, d
  ) GROUP BY cluster`;

const INERTIA = `
  SELECT sum(array_cosine_distance(r.embedding, ct.centroid))
  FROM requirements r JOIN assignments a USING (id) JOIN centroids ct USING (cluster)`;

const SILHOUETTE = (table, where = "") => `
  WITH asg AS (SELECT id, cluster FROM ${table} ${where}),
  d AS (
    SELECT a.id AS i, b.id AS j, array_cosine_distance(a.embedding, b.embedding) AS dist
    FROM requirements a, requirements b WHERE a.id <> b.id),
  m AS (SELECT d.i, aj.cluster AS cj, avg(dist) AS md FROM d JOIN asg aj ON aj.id = d.j GROUP BY d.i, aj.cluster),
  s AS (
    SELECT ai.id, max(md) FILTER (WHERE cj = ai.cluster) AS a, min(md) FILTER (WHERE cj <> ai.cluster) AS b
    FROM m JOIN asg ai ON ai.id = m.i GROUP BY ai.id)
  SELECT avg(CASE WHEN a IS NULL THEN 0 ELSE (b - a) / greatest(a, b) END) FROM s`;

async function kmeansRun(k, restart) {
  await initCentroids(k, restart);
  await run(`DELETE FROM assignments`);
  await run(`INSERT INTO assignments ${ASSIGN}`);
  let iterations = 0;
  for (; iterations < MAX_ITER; iterations++) {
    await run(`DELETE FROM centroids`);
    await run(UPDATE_CENTROIDS);
    const changed = await one(`
      SELECT count(*) FROM (${ASSIGN}) n JOIN assignments a USING (id) WHERE n.cluster <> a.cluster`);
    if (Number(changed) === 0) break;
    await run(`CREATE OR REPLACE TEMP TABLE next_assignments AS ${ASSIGN}`);
    await run(`DELETE FROM assignments`);
    await run(`INSERT INTO assignments SELECT * FROM next_assignments`);
  }
  const clusters = Number(await one(`SELECT count(DISTINCT cluster) FROM assignments`));
  if (clusters < k) return null; // a cluster ran empty
  return { iterations, inertia: Number(await one(INERTIA)) };
}

const candidates = [];
for (let k = K_RANGE[0]; k <= K_RANGE[1]; k++) {
  let best = null;
  for (let r = 0; r < RESTARTS; r++) {
    const res = await kmeansRun(k, r);
    if (!res) continue;
    const silhouette = Number(await one(SILHOUETTE("assignments")));
    await run(`INSERT INTO kmeans_runs VALUES (${k}, ${r}, ${res.iterations}, ${res.inertia}, ${silhouette})`);
    if (!best || res.inertia < best.inertia) {
      best = { ...res, silhouette };
      await run(`DELETE FROM assignments_by_k WHERE k = ${k}`);
      await run(`INSERT INTO assignments_by_k SELECT ${k}, id, cluster FROM assignments`);
    }
  }
  candidates.push({ k, ...best });
  console.log(`k=${k}  inertia=${best.inertia.toFixed(4)}  silhouette=${best.silhouette.toFixed(4)}`);
}
const chosen = candidates.reduce((a, b) => (b.silhouette > a.silhouette ? b : a));
console.log(`Chosen k=${chosen.k}`);

// Final result tables, clusters renumbered by their smallest requirement id
await run(`
  CREATE TABLE requirement_clusters AS
  SELECT id, dense_rank() OVER (ORDER BY first_id) - 1 AS cluster
  FROM (SELECT id, min(id) OVER (PARTITION BY cluster) AS first_id FROM assignments_by_k WHERE k = ${chosen.k})`);
await run(`DELETE FROM assignments`);
await run(`INSERT INTO assignments SELECT * FROM requirement_clusters`);
await run(`DELETE FROM centroids`);
await run(UPDATE_CENTROIDS);
await run(`CREATE TABLE clusters AS SELECT cluster, centroid FROM centroids`);

// Representative requirements: nearest neighbours of each centroid (vector search in the DB)
const representatives = await all(`
  SELECT cluster, list(id ORDER BY dist) AS ids FROM (
    SELECT rc.cluster, r.id, array_cosine_distance(r.embedding, c.centroid) AS dist,
           row_number() OVER (PARTITION BY rc.cluster ORDER BY array_cosine_distance(r.embedding, c.centroid)) AS rn
    FROM requirements r JOIN requirement_clusters rc USING (id) JOIN clusters c USING (cluster))
  WHERE rn <= 3 GROUP BY cluster`);

const rows = (await all(`
  SELECT r.id, r.text, r.embedding, rc.cluster
  FROM requirements r JOIN requirement_clusters rc USING (id) ORDER BY r.id`))
  .map((r) => ({ ...r, id: Number(r.id), cluster: Number(r.cluster), embedding: r.embedding.map(Number) }));
const labels = rows.map((r) => r.cluster);

// 4. Keywords per cluster (TF-IDF) as naming help
const STOP = new Set(("a an the and or of to in on for by with is are be can may must should will not no " +
  "it its this that those these as at from has have he she her his their they we our us also only one " +
  "each any all which who whether then there so e g into above out than more many other same own " +
  "already once order orders customer customers").split(" "));
const tokens = rows.map((r) => r.text.toLowerCase().match(/[a-z]{3,}/g).filter((t) => !STOP.has(t)));
const df = new Map();
tokens.forEach((ts) => new Set(ts).forEach((t) => df.set(t, (df.get(t) ?? 0) + 1)));
function keywords(c) {
  const tf = new Map();
  tokens.forEach((ts, i) => labels[i] === c && ts.forEach((t) => tf.set(t, (tf.get(t) ?? 0) + 1)));
  return [...tf].map(([t, f]) => [t, f * Math.log(rows.length / df.get(t))])
    .sort((a, b) => b[1] - a[1]).slice(0, 6).map(([t]) => t);
}

// Names are filled in by hand in data/cluster-names.json; create an empty template if missing
if (!fs.existsSync(namesFile)) {
  const template = Object.fromEntries(Array.from({ length: chosen.k }, (_, c) => [c, { name: "", description: "" }]));
  fs.writeFileSync(namesFile, JSON.stringify(template, null, 2) + "\n");
}
const names = JSON.parse(fs.readFileSync(namesFile, "utf8"));

const clusters = Array.from({ length: chosen.k }, (_, c) => {
  const members = rows.filter((r) => r.cluster === c);
  return {
    cluster: c,
    name: names[c]?.name || null,
    description: names[c]?.description || null,
    size: members.length,
    keywords: keywords(c),
    representative: representatives.find((r) => Number(r.cluster) === c).ids.map(Number),
    requirements: members.map(({ id, text }) => ({ id, text })),
  };
});

// 5. 2D projection (PCA via power iteration) for a scatter plot
const X = rows.map((r) => r.embedding);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
function pca2(data) {
  const mean = data[0].map((_, d) => data.reduce((s, x) => s + x[d], 0) / data.length);
  let centered = data.map((x) => x.map((v, d) => v - mean[d]));
  const comps = [];
  for (let p = 0; p < 2; p++) {
    let v = centered[0].map((_, d) => Math.sin(d + 1 + p));
    for (let it = 0; it < 200; it++) {
      const scores = centered.map((x) => dot(x, v));
      const nv = v.map((_, d) => centered.reduce((s, x, i) => s + x[d] * scores[i], 0));
      const norm = Math.hypot(...nv);
      v = nv.map((u) => u / norm);
    }
    comps.push(v);
    centered = centered.map((x) => { const s = dot(x, v); return x.map((u, d) => u - s * v[d]); });
  }
  return data.map((x) => comps.map((v) => dot(x.map((u, d) => u - mean[d]), v)));
}
const xy = pca2(X);
const COLORS = ["#2a78d6", "#d4570b", "#2f9e44", "#c2255c", "#7048e8", "#0c8599", "#e67700", "#5c7cfa", "#868e96", "#a61e4d"];
const W = 760, H = 520, PAD = 40;
const [xs, ys] = [xy.map((p) => p[0]), xy.map((p) => p[1])];
const sx = (v) => PAD + ((v - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs))) * (W - 2 * PAD - 200);
const sy = (v) => H - PAD - ((v - Math.min(...ys)) / (Math.max(...ys) - Math.min(...ys))) * (H - 2 * PAD);
const svg = [
  `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="sans-serif" font-size="11">`,
  `<rect width="100%" height="100%" fill="white"/>`,
  `<text x="${PAD}" y="22" font-size="14" font-weight="bold">EarlyBird requirement clusters (PCA 2D, k=${chosen.k})</text>`,
  ...rows.map((r, i) =>
    `<circle cx="${sx(xy[i][0]).toFixed(1)}" cy="${sy(xy[i][1]).toFixed(1)}" r="6" fill="${COLORS[labels[i]]}"/>` +
    `<text x="${(sx(xy[i][0]) + 8).toFixed(1)}" y="${(sy(xy[i][1]) + 4).toFixed(1)}" fill="#444">${r.id}</text>`),
  ...clusters.map((c, i) =>
    `<circle cx="${W - 200}" cy="${60 + i * 22}" r="6" fill="${COLORS[i]}"/>` +
    `<text x="${W - 188}" y="${64 + i * 22}">Cluster ${c.cluster}${c.name ? `: ${c.name}` : ""}</text>`),
  `</svg>`,
].join("\n");

// 6. Output
fs.writeFileSync(path.join(OUT, "embeddings.json"), JSON.stringify(rows.map(({ id, embedding }) => ({ id, vector: embedding }))));
fs.writeFileSync(path.join(OUT, "clusters.json"), JSON.stringify({
  model: MODEL, database: "DuckDB", k: chosen.k,
  silhouetteByK: Object.fromEntries(candidates.map((c) => [c.k, +c.silhouette.toFixed(4)])),
  clusters,
}, null, 2));
fs.writeFileSync(path.join(OUT, "clusters.svg"), svg);
await run(`DROP TABLE assignments_by_k`);
db.closeSync();

for (const c of clusters) {
  console.log(`\nCluster ${c.cluster}${c.name ? ` – ${c.name}` : ""} (${c.size}) [${c.keywords.join(", ")}]`);
  c.requirements.forEach((r) => console.log(`  ${String(r.id).padStart(2)}  ${r.text.slice(0, 110)}`));
}
console.log("\nWrote output/clusters.json, output/clusters.svg, output/embeddings.json");
