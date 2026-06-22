# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gensim==4.4.0",
#     "numpy==2.4.6",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import gensim.downloader as api

    return (api,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download
    """)
    return


@app.cell
def _(api):
    print("Loading word vectors (this may take a minute on the first run)...")
    # Download and load a lightweight 100-dimensional GloVe model
    model = api.load("glove-wiki-gigaword-100")
    # model = api.load("glove-wiki-gigaword-50")
    print("Model loaded successfully!\n")
    return (model,)


@app.cell
def _(model):
    def demonstrate_vector_math(positive_words, negative_words):
        """
        Computes: (positive_words[0] + positive_words[1]...) - (negative_words[0]...)
        """
        # Formula: positive=[A, C], negative=[B] -> translates to A - B + C
        result = model.most_similar(positive=positive_words, negative=negative_words, topn=1)
    
        pos_str = " + ".join(positive_words)
        neg_str = " - ".join(negative_words)
        equation = f"{pos_str} - {neg_str}" if neg_str else pos_str
    
        print(f"Equation: {equation}")
        print(f"Result:   {result[0][0]} (Confidence: {result[0][1]:.4f})\n")

    return (demonstrate_vector_math,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo
    """)
    return


@app.cell
def _(demonstrate_vector_math):
    # The Classic: King - Man + Woman = Queen
    # Rearranged for Gensim: King + Woman - Man
    demonstrate_vector_math(positive_words=['king', 'woman'], negative_words=['man'])

    # Geopolitics: Paris - France + Italy = Rome
    # Rearranged for Gensim: Paris + Italy - France
    demonstrate_vector_math(positive_words=['paris', 'italy'], negative_words=['france'])

    # WWII
    demonstrate_vector_math(positive_words=['hitler', 'italy'], negative_words=['germany'])

    # Grammar Tense: Walking - Walked + Swam = Swimming
    # Rearranged for Gensim: walking + swam - walked
    demonstrate_vector_math(positive_words=['walking', 'swam'], negative_words=['walked'])

    # Branding/Products: iPhone - Apple + Google = Android / Pixel
    # Rearranged for Gensim: iphone + google - apple
    demonstrate_vector_math(positive_words=['iphone', 'google'], negative_words=['apple'])

    # Pure fun
    demonstrate_vector_math(positive_words=['banker', 'suit'], negative_words=['tie'])
    demonstrate_vector_math(positive_words=['banker', 'computer'], negative_words=['tie'])
    demonstrate_vector_math(positive_words=['banker', 'suit'], negative_words=['wealthy'])
    return


@app.cell
def _():
    return


@app.cell
def _(model):
    import numpy as np

    def manual_cosine_similarity(v1, v2):
        """
        Calculates the cosine similarity between two vectors:
        dot_product(v1, v2) / (norm(v1) * norm(v2))
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
    
        # Avoid division by zero just in case
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)

    def raw_most_similar(pos_words, neg_words, top_n=3):
        # 1. Calculate the target vector destination: Target = King + Woman - Man
        target_vector = np.zeros(model.vector_size)
    
        for word in pos_words:
            target_vector += model[word]
        for word in neg_words:
            target_vector -= model[word]
        
        # 2. Track the best matches
        scores = []
    
        # We combine inputs so we can filter them out later
        input_words = set(pos_words + neg_words)
    
        # 3. The Raw Loop: Iterate through every single word in the entire vocabulary

        for word in model.index_to_key:
            # Skip the words we used to create the equation
            if word in input_words:
                continue
            
            # Get the vector for the current word in the loop
            current_vector = model[word]
        
            # Calculate similarity between our target point and this vocabulary word
            sim_score = manual_cosine_similarity(target_vector, current_vector)
        
            # Store the word and its score
            scores.append((word, sim_score))
        
        # 4. Sort the scores from highest (closest to 1.0) to lowest
        scores.sort(key=lambda x: x[1], reverse=True)
    
        # Return the top N results
        return scores[:top_n]

    return (raw_most_similar,)


@app.cell
def _(raw_most_similar):
    # --- Test our raw loop ---
    pos = ['king', 'woman']
    neg = ['man']

    # pos = ['paris', 'italy']
    # neg = ['france']

    print(f"Calculating: {pos[0]} - {neg[0]} + {pos[1]}...")
    results = raw_most_similar(pos_words=pos, neg_words=neg, top_n=4)

    print("\nTop 4 Matches from manual loop:")
    for rank, (word, score) in enumerate(results, 1):
        print(f"{rank}. {word}: {score:.4f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
