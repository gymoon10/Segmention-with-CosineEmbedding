from models.ERFNet_Semantic_Ori import ERFNet_Semantic_Ori
from models.ERFNet_Semantic_Embedding import ERFNet_Semantic_Embedding

def get_model(name, model_opts):

    if name == "ERFNet_Semantic_Ori":
        model = ERFNet_Semantic_Ori(num_classes=3)
        return model

    elif name == "ERFNet_Semantic_Embedding":
        model = ERFNet_Semantic_Embedding(num_classes=3)
        return model


    else:
        raise RuntimeError("model \"{}\" not available".format(name))