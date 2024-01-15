# Chroma implementation for Python

## Objective
Understand vector databases by implementing Chroma locally. My core objective of using  
open-source models and methods is still valid. I wish to use a locally trained model to 
embed my data.

## About Chroma
Chroma is an open-source embedding database. With several ways to get to the embedding,
this is one of the most popular databases to learn and test your learnings before moving them 
into a production environment. You can find all the information and a quick tutorial on how to
implement a project using Chroma [here](https://docs.trychroma.com/).

## Embedding function
This is the model we are going to use to create our embeddings. At the time of creating a 
collection, if no function is specified, it would default to the "Sentence Transformer". 

Chroma comes with lightweight wrappers for various embedding providers. "OpenAI", "Google PaLM", 
and "HuggingFace" are some of the more popular ones. In this example, I will be creating my 
custom embedding function. 

I am using a pre-trained model for this example. But this model can be replaced with a model 
that has been trained and downloaded onto a local folder. 

### Explanation of the code
1. Create a custom embedding function class that inherits from "EmbeddingFunction"

```
class MyEmbeddingFunction(EmbeddingFunction):
```

2. We will override the "__init__" function by initializing the tokenizer and the 
model we will be using.

```
def __init__(self):
    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    self.model = AutoModel.from_pretrained('bert-base-cased')
```

3. We will also need to define the "__call__" function that will be executed each 
time the embedding function is used

```
def __call__(self, input: Documents) -> Embeddings:
        # Embedding the documents
        list_emb = []

        for doc in input:
            tokens = self.tokenizer(doc,
                                    padding='max_length',
                                    return_tensors='pt')
            output = self.model(**tokens)
            embeddins = output['last_hidden_state'][0].detach().flatten().tolist()
            list_emb.append(embeddins)
        return list_emb
```

"Documents" is of type "List[str]". We will loop through this list of documents and 
for each of these documents, we will 

- Tokenize the document using the tokenizer
- Pass the tokens to the model
- Extract the data from the hidden layer. This is the layer just before the
  classification.
- "Detach" it from the tensor, "flatten" it to a one-dimensional array, and then
  convert the output to a list. 
- Append the output list back into a list that is the variable that we will return.


In the last step, we will initialize our custom embedding function. This is the function we will use when calling the "create_or_get_collection" function.

```
# Initializing my custom embedding function
st_ef = MyEmbeddingFunction()
```

The full code can be found in the notebook file in this folder.


