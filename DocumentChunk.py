import json

class DocumentChunks_With_Embeddings:
    def __init__(self, embeddings, text_chunks, filename):
        self.embeddings = embeddings
        self.text_chunks = text_chunks
        self.filename = filename
        
    # def add(self, new_embedding, new_chunk, new_filename):
#        new_document_chunk = DocumentChunk(new_embedding, new_chunk, new_filename)
        # Add the new document chunk to the existing list or data structure
        # For example, if you want to store the document chunks in a list:
 #       self.document_chunks.append(new_document_chunk)
        def toJSON(self):
            return json.dumps(
                self,
                default=lambda o: o.__dict__, 
                sort_keys=True,
                indent=4)
