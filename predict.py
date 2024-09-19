# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from transformers import AutoModel
from PIL import Image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model =AutoModel.from_pretrained(pretrained_model_name_or_path="/weights",trust_remote_code=True)
    def predict(
        self,
        image_url: Path = Input(description="url of images to embed"),
        texts: str = Input(
            description="texts to embed"
        ),
    ) -> dict:
        """Run a single prediction on the model"""
        image = Image.open(image_url)
        text_embeddings  =  self.model.encode_text([texts])
        image_embeddings = self.model.encode_image([image])
        result = {"text_embeddings":text_embeddings.tolist(),"image_embeddings":image_embeddings.tolist()}
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        return result
