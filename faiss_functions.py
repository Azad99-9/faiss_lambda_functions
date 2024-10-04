import faiss
import numpy as np
import json

# Initialize the FAISS index globally to preserve the state between invocations
index = None

def load_vector_database(event, context):
    global index
    
    # Parse the request body to get the vectors
    body = json.loads(event['body'])
    
    # Check if 'vectors' key is present in the body
    if 'vectors' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': "Missing 'vectors' key in request body."})
        }
    
    # Convert the list of vectors to a NumPy array
    try:
        # Assuming vectors is a 2D array where each inner array is a vector
        vectors = np.array(body['vectors']).astype('float32')
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f"Invalid vectors format: {str(e)}"})
        }

    # Initialize the FAISS index if it hasn't been created yet
    if index is None:
        # Create a new index with the dimensionality of the first vector
        d = vectors.shape[1]  # Dimensionality inferred from the first vector
        index = faiss.IndexFlatL2(d)  # Using L2 distance index

    # Add the vectors to the FAISS index
    index.add(vectors)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f"Successfully added {len(vectors)} vectors to the index.",
            'current_index_size': index.ntotal  # Returns the current number of vectors in the index
        })
    }

def search_vector(event, context):
    global index
    
    # Parse the request body to get the query vector
    body = json.loads(event['body'])
    
    # Check if 'query_vector' key is present in the body
    if 'query_vector' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': "Missing 'query_vector' key in request body."})
        }

    # Convert the query vector to a NumPy array
    try:
        query_vector = np.array(body['query_vector']).astype('float32')
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be a 1D array.")
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f"Invalid query vector format: {str(e)}"})
        }

    # Perform the search in the FAISS index
    k = 5  # Number of nearest neighbors to search
    distances, indices = index.search(query_vector.reshape(1, -1), k)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'distances': distances.tolist(),
            'indices': indices.tolist()
        })
    }
