# NUMERICAL PATTERN IN VARIOUS SENTIMENTS

This paper aims in finding numerical patterns in various sentences. This project aims in giving a range of numerical value given a sentence containing complex emotions. Then, the numerical values can be further used for various other data analytics.

## Technical Description of Project

1. **Web Scraping**:
    - We are using web-scraping techniques to get various posts on Reddit. The dataset used in this project is mainly from Reddit.

2. **Praw Service**:
    - We use the “Praw” service to scrape Reddit data from various relevant subreddits as shown in the code. We create a service (API) with relevant credentials to access the service.

3. **Spacy Library**:
    - We use a library called “spacy” to create our graph. We load a trained NLP model which provides various functionalities. We mainly use two functionalities: Tokenization and POS tags.

4. **Functions**:
    - We create two functions, `get_entities` and `get_relations`. The `get_entities` function gives us nouns, and the `get_relations` function gives us verbs.

5. **Knowledge Graph**:
    - We then create a Knowledge Graph where nodes are nouns and edges are verbs using the “networkx” library. We construct a graph for posts representing each emotion and determine the numerical range for each pattern.

6. **Mathematical Formula**:

    \[
    \text{Mathematical Formula:} - \frac{\text{Sigmoid}\left(\sum_{i=1}^{n} n_i \cdot \lambda_i\right)}{N} + \frac{\sum_{i=1}^{n} \alpha_i \cdot (\text{total number of neighbours})}{N} + F
    \]

    - Eigen Vector Centrality
    - Katz-Centrality
    - \(\lambda = \alpha =\)
    - \(n_i = \text{node}\)
    - \(N = \text{Total number of nodes}\)
    - \(F = \text{Estrada index}\)

7. **Testing**:
    - Once the range has been decided, we test it again with various other sentences and found that similar emotions give out similar numerical values.

