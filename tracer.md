# Tracer Bullet

## Goal
Thinnest possible slice through every layer of the system that produces a real recommendation from a real user interaction.

## Flow
1. **Frontend** (`app/home.html`): A "Recommend" button. On click, fetches `GET /recommend` from the backend.
2. **Backend** (`app/api.py`): A FastAPI app with CORS middleware. On startup, creates a `Tracer` instance with `test_books` (a hardcoded DataFrame from `tests/test_tracer.py`). The `/recommend` endpoint calls `model.recommend()` and casts `rating_5` to `int()` (needed because FastAPI can't serialize `numpy.int64`).
3. **Model** (`src/main.py`): `Tracer.recommend()` finds the row with the max `ratings_5` value using `idxmax()`, then returns a dict with `title`, `authors`, and `rating_5`.
4. **Frontend**: Displays "`{title}` by `{authors}` with `{rating_5}` 5 star ratings."

## Test
`tests/test_tracer.py` defines a 4-book `test_books` DataFrame and asserts that `recommend()` returns Percy Jackson (highest `ratings_5` = 30).

## Resolved Questions
- **When does data loading happen?** On startup. The `Tracer` is instantiated once at module level in `api.py`, not per request.
- **Which component to build first?** The model.
- **Where does the data come from?** Currently from a hardcoded `test_books` DataFrame. The `pd.read_csv('data/books.csv')` line in `api.py` is commented out for later use.
