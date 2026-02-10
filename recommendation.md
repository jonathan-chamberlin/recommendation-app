# Content Discovery Engine — Technical Plan

## One-Line Problem Statement

*Given a few books you rate, discover books you'll love — with explanations for every recommendation.*

## Why This Project

Your portfolio has strong RL projects (agent-organism, lunar-lander) and data analysis (Uber trips). All three share the same blind spots:

- No live URL anyone can click
- No supervised learning or deep learning for prediction
- No recommendation or personalization (39% of target job postings)
- No deployment (no API, no Docker, no cloud)
- No PyTorch (78% of job postings want it)

This project fills every gap in a single build.

---


## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    DEMO (React + shadcn/Tailwind)         │
│  Onboarding → Rate books (1-5 stars) → Get recs → Feedback│
└──────────────────────┬────────────────────────────────────┘
                       │ REST API
┌──────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                           │
│  /recommend  /feedback  /onboarding  /health  /ab-test    │
└──────┬──────────────────────┬─────────────────────────────┘
       │                      │
┌──────▼──────────────┐  ┌───▼─────────────────────────────┐
│  Two-Tower Model     │  │  Content Feature Branch          │
│  (PyTorch)           │  │  (for explainability + cold-start)│
│                      │  │                                   │
│  User Tower:         │  │  Genre, topic, and description    │
│    User interaction  │  │  features augment CF scores with  │
│    history → user    │  │  human-readable explanations      │
│    embedding         │  │                                   │
│  Item Tower:         │  │  "Recommended because: Science    │
│    Item features →   │  │   Fiction + rated 4.2 avg by      │
│    item embedding    │  │   similar users"                  │
│                      │  │                                   │
│  Score: dot product  │  └───────────────────────────────────┘
│  of user & item      │
│  embeddings          │
└──────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────┐
│  Goodreads-10K Dataset                                    │
│  10,000 books · 50,000 users · 6,000,000 ratings          │
│  Train (80%) / Validation (10%) / Test (10%)              │
└──────────────────────────────────────────────────────────┘
```

---

## Design Decisions & Rationale

### 1. Two-Tower Collaborative Filtering (Primary)

**Choice:** Two-Tower model — separate neural networks for user and item, merged via dot product.

**Why:** This is what YouTube and Google use in production for candidate generation. It's the most interview-relevant CF architecture. The two towers can be served independently — item embeddings are precomputed, only the user tower runs at inference time.

**Tradeoff accepted:** More complex than Matrix Factorization or NCF. Harder to debug. If training is unstable, NCF is the fallback (80% of the impressiveness at 50% of the effort).

**What you must be able to explain from memory:**
- Why two separate towers instead of one network?
- How does the dot product scoring work?
- Why is this architecture efficient at scale (precomputed item embeddings)?
- What's the difference between candidate generation and re-ranking?

### 2. Content Feature Hybrid (For Explainability)

**Choice:** Augment CF scores with content metadata (genre, topic, description features) to generate human-readable explanations.

**Why:** Pure CF embeddings are latent factors — no human-readable meaning. Adding content features lets you say *"Recommended because: Science Fiction + rated 4.2 avg by similar users"* instead of just showing a confidence score.

**Tradeoff accepted:** Hybrid is more complex than pure CF. The content branch adds feature engineering, a second model component, and a score combination strategy to tune.

**Implementation:** CF produces a relevance score. Content branch produces explanation metadata (shared genres, similar rated items). The ranking uses CF scores; the explanation uses content features. They don't need to be a single model.

### 3. Explicit 1-5 Star Ratings (Training and Demo)

**Choice:** Use explicit ratings in both the training data (Goodreads) and the demo interface.

**Why:** Eliminates the train/inference feedback mismatch. Goodreads has 1-5 star ratings. The demo collects 1-5 star ratings. The model sees the same signal in both contexts.

**Tradeoff accepted:** Star ratings are a worse UX than thumbs up/down (more friction per interaction). But the consistency between training and serving is more important for a portfolio project than UX optimization.

### 4. React Frontend with shadcn/Tailwind Template

**Choice:** React with a pre-built component library, not Streamlit.

**Why:** Streamlit signals "data science prototype." React signals "this person can ship products." Using shadcn/Tailwind gets you professional aesthetics in 3-4 days instead of 2 weeks of raw React.

**Tradeoff accepted:** Frontend work doesn't demonstrate ML skill. If the project falls behind schedule, the frontend is the first thing to simplify (fallback: Streamlit).

### 5. Evaluation: Offline Metrics + User Studies + Simulated A/B Test

**Choice:** Three layers of evaluation rigor.

**Why:**
- **Offline metrics** (NDCG@10, Hit Rate@10, MRR) prove the model works on held-out data. Standard methodology.
- **User studies** (5-10 friends test the demo and rate quality) prove the model works for real humans. Shows you think about users, not just numbers.
- **Simulated A/B test** (compare two model versions on held-out user groups) proves you understand experimentation. A/B testing appears in 42% of target job postings.

**Tradeoff accepted:** User studies require coordination time. Simulated A/B adds implementation scope. If behind schedule, cut the A/B simulation first (the concept can be explained verbally).

### 6. Books Only (YouTube as Stretch Goal)

**Choice:** Build the core system for book recommendations using Goodreads data. YouTube cross-domain is a stretch goal only if the core is solid.

**Why:** Collaborative filtering requires user-item interaction data. Goodreads has 6M real ratings. YouTube has zero. Adding YouTube requires a content-based branch (sentence-transformer embeddings) that allows cross-domain transfer — this is architecturally possible but adds significant scope and introduces the riskiest assumption in the project (do book preferences transfer to video preferences?).

**Stretch plan:** If core book system is complete and polished by end of Phase D, add a content-based branch that encodes YouTube video descriptions into the same space as book descriptions. The hybrid system uses CF scores for books (where you have interaction data) and content-based scores for YouTube (where you only have metadata). This is how real production systems handle cold-start items.

### 7. Scale: Verbal Explanation Only

**Choice:** Don't build scalability infrastructure (FAISS, candidate generation pipeline). Be ready to explain it in an interview.

**Why:** 10K books is a toy dataset. Building FAISS indexing would take 2 days but wouldn't meaningfully improve the demo. The time is better spent on model quality and evaluation rigor.

**What you must be able to explain from memory:**
- "At 10K items, brute-force dot product is fine. At 100M items, I'd add a candidate generation stage using approximate nearest neighbors (FAISS or ScaNN) to retrieve the top 1000 candidates in sub-millisecond time, then re-rank with the full model."
- "The Two-Tower architecture supports this naturally — item embeddings are precomputed and indexed. Only the user tower runs online."
- Understand ANN algorithms at a high level (LSH, HNSW, IVF)

---

## Dataset

**Goodreads-10K** ([github.com/zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k))

| Property | Value |
|----------|-------|
| Books | 10,000 |
| Users | 53,424 |
| Ratings | 5,976,479 |
| Rating scale | 1-5 stars |
| Metadata | Title, author, genre tags, description, cover image URL, average rating, publication year |
| Split | 80% train / 10% validation / 10% test (by user, not by rating) |

**Why this dataset:** Clean, well-documented, real human ratings, rich metadata for the content feature branch. Large enough to train a meaningful model, small enough to iterate fast.

**Important:** Split by user, not by rating. Hold out entire users for the test set, not random ratings from all users. This tests generalization to new users, which is what the demo does.

---

## Evaluation Plan

### Offline Metrics (Required)

| Metric | What It Measures | Baseline |
|--------|-----------------|----------|
| NDCG@10 | Are top 10 recs in the right order? | Random, Popularity, User-average |
| Hit Rate@10 | Is at least one held-out liked item in top 10? | Random, Popularity |
| MRR | How high does the first correct rec rank? | Random, Popularity |

Report all metrics for your model AND all baselines. Show the delta.

### User Studies

- Recruit 5-10 friends/classmates
- Each person rates 10-15 books in the onboarding flow
- Rate the quality of recommendations 1-5 ("Would you actually read this?")
- Collect qualitative feedback ("Why did you rate this rec low?")
- Report: average recommendation quality score, common failure modes

### Simulated A/B Test (Cut if Behind)

- Split held-out test users into two groups (A and B)
- Group A gets recommendations from the Two-Tower model
- Group B gets recommendations from the popularity baseline
- Compare NDCG@10 between groups
- Report: effect size, confidence interval, statistical significance (t-test or Mann-Whitney U)
- Show this in the README as an A/B test result with a clear visualization

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| ML framework | PyTorch | 78% of target job postings. Deep learning standard. |
| Data processing | pandas, NumPy | Data loading, feature engineering |
| Text embeddings | sentence-transformers (stretch goal only) | For YouTube cross-domain content-based branch |
| Backend | FastAPI | Async, fast, auto-generated API docs, type-safe |
| Frontend | React + shadcn/ui + Tailwind CSS | Professional look with minimal custom CSS |
| Config | PyYAML | No hardcoded hyperparameters |
| Experiment tracking | Weights & Biases | Log training metrics, hyperparameters, model versions |
| Testing | pytest | Unit tests for data transforms, model inference, API endpoints |
| Containerization | Docker | Reproducible deployment |
| Deployment | Railway or Render | Free tier, easy Docker deployment, live URL |

---

## Phased Plan

### Phase A — Data Pipeline + Infrastructure

**Goal:** Repository set up, data loaded and explored, training infrastructure ready.

**Tasks:**
- [ ] Create repo with demo-guidelines folder structure (`src/`, `tests/`, `configs/`, `scripts/`, `app/`)
- [ ] Download Goodreads-10K dataset, explore distributions (rating counts, user activity, genre spread)
- [ ] Implement data loading pipeline: PyTorch `Dataset` and `DataLoader` for user-item interactions
- [ ] Implement train/validation/test split BY USER (not by rating)
- [ ] Set up YAML config for all hyperparameters (embedding dim, learning rate, batch size, etc.)
- [ ] Set up Weights & Biases experiment tracking
- [ ] Implement baseline models: Random recommender, Popularity-based recommender
- [ ] Evaluate baselines on test set (NDCG@10, Hit Rate@10, MRR)
- [ ] Set random seeds for reproducibility

**Milestone:** Baselines evaluated and logged. You can explain the dataset, the split strategy, and why the baselines perform the way they do.

**Recursive gap filling targets:** PyTorch Dataset/DataLoader, NDCG metric computation, train/val/test split strategies for recommendation

---

### Phase B — Two-Tower Model

**Goal:** Core ML model trained and beating baselines.

**Tasks:**
- [ ] Implement User Tower: takes user interaction history (item IDs + ratings) → user embedding vector
- [ ] Implement Item Tower: takes item features (ID, genre tags, metadata) → item embedding vector
- [ ] Implement scoring: dot product of user embedding and item embedding → predicted relevance
- [ ] Implement training loop with BPR loss (pairwise ranking loss)
  - Positive samples: items the user rated highly (4-5 stars)
  - Negative samples: items the user hasn't interacted with
  - Negative sampling strategy: random with popularity weighting
- [ ] Train on Goodreads training set, validate on validation set
- [ ] Evaluate on test set: NDCG@10, Hit Rate@10, MRR
- [ ] Compare against baselines — must show meaningful improvement
- [ ] Hyperparameter tuning: embedding dimension, learning rate, batch size, negative sampling ratio
- [ ] Log all experiments to W&B with metrics, hyperparameters, and model checkpoints
- [ ] Implement model saving and loading (versioned checkpoints)
- [ ] If Two-Tower training is unstable: fallback to NCF (simpler architecture, same evaluation)

**Milestone:** Model beats all baselines by a meaningful margin. Training is stable and reproducible. You can draw the Two-Tower architecture on a whiteboard and explain every component.

**Recursive gap filling targets:** Embedding layers in PyTorch, BPR loss derivation and intuition, negative sampling strategies, Two-Tower vs NCF vs Matrix Factorization tradeoffs

---

### Phase C — Explainability + Content Feature Hybrid

**Goal:** Recommendations come with human-readable explanations.

**Tasks:**
- [ ] Extract content features from Goodreads metadata: genre tags, author, publication year, average rating, description keywords
- [ ] For each recommendation, compute explanation:
  - Find top-3 books from the user's liked set that are most similar (by genre overlap, author overlap, or embedding distance)
  - Extract shared attributes: "Science Fiction, rated 4.3 avg by similar users"
  - Format as: *"Recommended because you liked [Book X] and [Book Y]. Shared genres: Science Fiction, Philosophy."*
- [ ] Implement cold-start handling: for the demo onboarding (new user with 5-10 ratings), the model must still produce reasonable recommendations
  - Test: create synthetic users with only 5 ratings, evaluate recommendation quality
  - If cold-start performance is poor: add a popularity-weighted fallback for users with <10 ratings
- [ ] Add content features as additional input to the item tower (genre embeddings, description features)
  - Evaluate whether this improves NDCG vs. ID-only item tower
  - If no improvement: keep the content features for explainability only, don't add to the model

**Milestone:** Every recommendation comes with a 1-2 sentence explanation. Cold-start users (5-10 ratings) get reasonable recommendations. You can explain the explainability approach and its limitations.

**Recursive gap filling targets:** Feature engineering for recommendation, content-based vs collaborative filtering tradeoffs, cold-start problem and solutions

---

### Phase D — API + Frontend + Demo

**Goal:** Live, interactive demo accessible via URL.

**Tasks:**
- [ ] Build FastAPI backend:
  - `POST /onboarding` — receive a list of book ratings, return initial recommendations
  - `POST /feedback` — receive updated rating, return re-ranked recommendations
  - `GET /recommend/{user_id}` — return top-K recommendations with explanations and confidence scores
  - `GET /health` — health check with model version, dataset version
  - Input validation, error handling, edge cases (empty history, invalid book IDs)
  - Log inference latency
- [ ] Build React frontend:
  - **Onboarding screen:** Grid of ~30 diverse books with cover images. User rates 5-10 books (1-5 stars). "Get Recommendations" button.
  - **Results screen:** Ranked list of recommended books. Each shows: cover, title, author, predicted rating, explanation text, confidence score.
  - **Feedback:** User can rate any recommended book. "Refresh" button re-ranks based on new ratings.
  - **Model info panel:** Dataset version, model version, evaluation metrics, commit hash.
  - Use shadcn/ui components and Tailwind for styling
- [ ] "Run example" button with pre-filled ratings for instant demo
- [ ] Handle edge cases in UI: loading states, error messages, empty results
- [ ] Fast inference (target: <500ms per recommendation request)

**Milestone:** Demo is functional end-to-end locally. Someone can open it, rate books, get recommendations with explanations, and give feedback that updates results.

**Fallback:** If React is taking too long and eating into ML time, fall back to Streamlit. A working Streamlit demo is better than an unfinished React app.

**Recursive gap filling targets:** FastAPI request/response patterns, React state management basics, API design for ML inference

---

### Phase E — Evaluation, Testing, Deploy, Document

**Goal:** Production-grade evaluation, testing, deployment, and documentation.

**Tasks:**

**Evaluation:**
- [ ] Run final offline evaluation on test set, report all metrics vs. all baselines
- [ ] Conduct user studies: recruit 5-10 people, have them use the demo, collect rating quality scores and qualitative feedback
- [ ] Implement simulated A/B test:
  - Split test users into Group A (Two-Tower model) and Group B (Popularity baseline)
  - Compute NDCG@10 for each group
  - Report effect size, 95% confidence interval, p-value (t-test or Mann-Whitney U)
  - Create visualization: bar chart or box plot comparing groups
- [ ] Write evaluation section in README with all results, baselines, and A/B test output

**Testing:**
- [ ] Unit tests for data transforms (rating normalization, feature encoding)
- [ ] Unit tests for model inference (deterministic output given fixed input)
- [ ] Unit tests for API endpoints (valid requests, invalid requests, edge cases)
- [ ] Sanity checks: same input → same output (reproducibility)
- [ ] Edge case tests: user with 1 rating, user who rates everything 5 stars, user who rates everything 1 star

**Deployment:**
- [ ] Dockerize backend and frontend
- [ ] Deploy to Railway or Render (free tier)
- [ ] Verify live URL works end-to-end
- [ ] Add model metadata to `/health` endpoint (version, timestamp, commit hash)

**Documentation:**
- [ ] README: problem statement, architecture diagram, model explanation, setup instructions
- [ ] Design tradeoffs section: why Two-Tower over NCF, why explicit ratings, why books-only, what you'd do differently
- [ ] Evaluation results with visualizations
- [ ] Known limitations
- [ ] Next steps / roadmap (including YouTube stretch goal)
- [ ] Record 30-60 second demo video or GIF
- [ ] Run through entire demo-guidelines.md checklist

**Milestone:** Demo is live at a URL. README is complete. Evaluation results are documented. You can explain the entire project from memory in 3-5 minutes.

---

### Phase F (Stretch) — YouTube Cross-Domain

**Prerequisite:** Phases A-E are complete and polished. Book recommendation engine is solid.

**Goal:** Extend the system to recommend YouTube videos alongside books using a content-based hybrid branch.

**Tasks:**
- [ ] Collect YouTube video metadata: titles, descriptions, categories, thumbnail URLs (~5K videos across diverse topics)
  - Source: YouTube Data API or Kaggle dataset
- [ ] Encode all item descriptions (books + videos) with sentence-transformers (`all-MiniLM-L6-v2`) into a shared embedding space
- [ ] Add content-based scoring branch:
  - For items with CF interaction data (books): use Two-Tower CF score as primary, content features for explanation
  - For items without interaction data (YouTube): use content-based similarity to the user's rated book embeddings
- [ ] Combine scores: CF score (when available) + content-based score (always available), weighted by confidence
- [ ] Update demo to show mixed results: books with CF-backed recommendations, YouTube videos with content-based recommendations
- [ ] Evaluate: report content-based metrics separately from CF metrics. Be transparent about which items use which method.
- [ ] Update README with cross-domain architecture diagram and honest discussion of transfer assumptions

**Why this is a stretch goal, not core:**
- CF needs user-item interaction data. YouTube has none. The YouTube recommendations are purely content-based (text similarity), which is weaker than CF.
- The assumption that book preferences transfer to video preferences via shared text embeddings is unproven. It might work for "Sapiens → Kurzgesagt" but fail for fiction → entertainment.
- Shipping a mediocre cross-domain system is worse than shipping an excellent single-domain system.

---

## Scope Management

**Total budget:** ~90-108 hours (9 hrs/week × 10-12 weeks)

**If behind schedule, cut in this order:**
1. **Phase F (YouTube stretch)** — cut entirely, no loss to core demo
2. **Simulated A/B test** — explain the concept verbally instead of implementing it
3. **React frontend** — fall back to Streamlit (saves ~1-2 weeks)
4. **User studies** — rely on offline metrics only

**Never cut:**
- Two-Tower model (the core ML)
- Offline evaluation with baselines (proves it works)
- Live deployment with URL (Gabriel's rule #3)
- README with architecture diagram and design tradeoffs

---

## What You Must Explain From Memory

After building this project, you should be able to answer all of these without notes:

**Model:**
- What is the Two-Tower architecture and why did you choose it?
- How does BPR loss work? Why pairwise ranking instead of pointwise MSE?
- What does each tower learn? What do the embeddings represent?
- How do you handle negative sampling? Why does the strategy matter?
- What's the cold-start problem and how did you address it?

**Evaluation:**
- What metrics did you use and why?
- What baselines did you compare against?
- How much did your model improve over popularity-based recommendations?
- What did the user studies reveal? Any surprising failure modes?
- How did you design the simulated A/B test?

**System Design:**
- How does this scale to 100M items? (FAISS, candidate gen + re-ranking)
- Why separate towers? (Precomputed item embeddings, efficient serving)
- What's the inference latency? What's the bottleneck?
- How would you add real-time updates in production?

**Tradeoffs:**
- Why collaborative filtering over content-based?
- Why explicit ratings over implicit feedback?
- Why books only? What would cross-domain require?
- What would you do differently with more time/data?

---

## Interview Story (3-5 Minutes)

*Practice this until you can deliver it naturally:*

**Problem:** "I built a book recommendation engine that takes a few ratings from a new user and recommends books they'll enjoy, with explanations for every recommendation."

**Approach:** "I used a Two-Tower collaborative filtering model trained on 6 million real ratings from the Goodreads dataset. The user tower learns a preference embedding from interaction history. The item tower learns a content embedding from book metadata. Scoring is a dot product — efficient because item embeddings are precomputed."

**Explainability:** "Pure CF gives opaque scores, so I added a content feature branch that extracts shared genres and similar books to generate explanations like 'Recommended because you liked Sapiens — shared genres: Science, History.'"

**Evaluation:** "I evaluated with NDCG@10 and Hit Rate@10 against random and popularity baselines. The Two-Tower model improved NDCG by [X]% over popularity. I also ran user studies with [N] people and a simulated A/B test showing statistically significant improvement."

**What I'd do differently:** "With more time, I'd add a content-based branch for cross-domain recommendations (YouTube videos), implement FAISS for approximate nearest neighbor search at scale, and move from explicit to implicit feedback signals."
