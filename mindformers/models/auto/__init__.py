"""init auto"""
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
    AutoModelWithLMHead
)
from .tokenization_auto import AutoTokenizer
from .image_processing_auto import AutoImageProcessor
