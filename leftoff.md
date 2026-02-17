2/16/2026
I have the tracer bullet. What to build next? Options include the model, frontend, fast api, deployment. Clearly the model is first thing to improve. I am using a two tower model. That means that we use both the text from the recommendations and the relationships between the data to determine ranking. 
- Users who like book A and B usually like book C. So if a customer likes books A and B, we should recommend book C.
- If a user like a book that contain the same words, or more speciifclaly are closer in a higher dimensional space, then the user is likely to like books that are near that first book.

Tower 1 takes in a user and outputs a vector representing the location of the user in the higher dimensional space.

Tower 1 is trained based off a bunch of pairs (user,book). Imagine user's vector is a house location, and the book's vector is a restaurant location. We have millions of pairs of (user, restaurant). We update the map so users who eat at a restaurant a lot, their house is put closer to it. And their house is moved further from restaurants they never eat at.
- 
Tower 2 takes in a book and outputs a vector representing the location of the book in higher dimensional space. 

Then we take the dot product between those two vectors and we get a scalar compatability score, representing how compatable the book is with the user. The book with the highest dot product is ranked the highest.




