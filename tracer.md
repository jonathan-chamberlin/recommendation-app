# Tracer Bullet

## Goal
Thinnest possible slice through every layer of the system that produces a real recommendation from a real user interaction.

## Flow
1. **Frontend**: Has a "Give Recommendation" button. When pressed, sends a request to the backend.
2. **Backend (API)**: Receives the request, asks the model for a book recommendation.
3. **Model**: Popularity baseline — finds the book with the most 5-star ratings (`rating_5`) from books.csv and returns it.
4. **Backend (API)**: Sends the result back to the frontend.
5. **Frontend**: Displays the book's name, author, and number of 5-star ratings.

## Resolved Questions
- **When does data loading happen?** On startup. The data and model are loaded once when the server starts, not per request.
- **Which component to build first?** The model.



- 
