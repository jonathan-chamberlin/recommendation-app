## Future Iteration: Personalized Recommendations
After the tracer bullet proves the plumbing works, add user-specific filtering:
1. Filter to books on the user's to-read list
2. If multiple remain, filter to books where the user has read a book with at least one matching tag
3. If still multiple, recommend the one with the highest average rating across users

## API
My tracer bullet model and personalized model should have the same json shape, because that would make it possible for the user to select which algorithm they use on the UI.