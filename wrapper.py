from torchvision.datasets import ImageFolder


class ImageFolderWrapper:
    def __init__(self, dataset: ImageFolder):
        self.dataset = dataset

    def __get_pydantic_core_schema__(self, handler):
        """
        Fornisce uno schema compatibile con Pydantic, trattando il dataset come un dizionario.
        """
        return handler.generate_schema(dict)

    def to_dict(self):
        """
        Converte l'oggetto `ImageFolder` in un dizionario serializzabile.
        """
        return {
            "root": self.dataset.root,
            "classes": self.dataset.classes,
            "class_to_idx": self.dataset.class_to_idx,
            "samples": len(self.dataset.samples),
        }
