# NUMERICAL PATTERN IN VARIOUS SENTIMENTS

This paper aims to find numerical patterns in various sentences. This project aims to provide a range of numerical values given a sentence containing complex emotions. These numerical values can then be used for various other data analytics.

## Technical Description of Project

1. **Web Scraping**:
    - We use web-scraping techniques to collect various posts on Reddit. The dataset used in this project is mainly from Reddit.

2. **Praw Service**:
    - We use the “Praw” service to scrape Reddit data from various relevant subreddits as shown in the code. We create a service (API) with relevant credentials to access the service.

3. **Spacy Library**:
    - We use a library called “spacy” to create our graph. We load a trained NLP model which provides various functionalities. We mainly use two functionalities: Tokenization and POS tags.

4. **Functions**:
    - We create two functions, `get_entities` and `get_relations`. The `get_entities` function gives us nouns, and the `get_relations` function gives us verbs.

5. **Knowledge Graph**:
    - We then create a Knowledge Graph where nodes are nouns and edges are verbs using the “networkx” library. We construct a graph for posts representing each emotion and determine the numerical range for each pattern.

6. **Mathematical Formula**:

    <img width="910" alt="image" src="https://github.com/user-attachments/assets/1a4eceae-8523-49f6-a3d8-aad2be6fc325">


    - Eigen Vector Centrality
    - Katz-Centrality
    - \(\lambda\), \(\alpha\) are constants
    - \(n_i\) is the node
    - \(N\) is the total number of nodes
    - \(F\) is the Estrada index

7. **Testing**:
    - Once the range has been decided, we test it again with various other sentences and found that similar emotions give out similar numerical values.
