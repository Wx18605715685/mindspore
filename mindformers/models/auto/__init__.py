"""init auto"""
from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from .image_processing_auto import AutoImageProcessor
from .processing_auto import AutoProcessor, PROCESSOR_MAPPING
from .configuration_auto import CONFIG_MAPPING, AutoConfig
from .modeling_auto import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedImageModeling,
    AutoModelForMaskGeneration,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForTextEncoding,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelWithLMHead,
    MODEL_FOR_CAUSAL_LM_MAPPING
)

__all__ = ["CONFIG_MAPPING", "AutoConfig", "TOKENIZER_MAPPING", "AutoTokenizer",
           "AutoImageProcessor", "AutoModel", "AutoModelForCausalLM", "AutoModelForImageClassification",
           "AutoModelForMaskedImageModeling", "AutoModelForMaskGeneration",
           "AutoModelForMultipleChoice", "AutoModelForPreTraining",
           "AutoModelForQuestionAnswering", "AutoModelForQuestionAnswering",
           "AutoModelForSeq2SeqLM", "AutoModelForTextEncoding", "AutoModelForTokenClassification",
           "AutoModelForVision2Seq", "AutoModelForVisualQuestionAnswering",
           "AutoModelForZeroShotImageClassification", "AutoModelWithLMHead",
           "MODEL_FOR_CAUSAL_LM_MAPPING", "AutoProcessor"]
