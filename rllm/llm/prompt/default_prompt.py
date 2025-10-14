DEFAULT_SCENARIO_CLASSIFICATION_TMPL = (  # noqa
    "The sceario is {scenario} with tabular data. "  # noqa
    "We first provide sample description contains columns and corresponding values, "  # noqa
    "and then we provide labels the sample may belong to.\n"  # noqa
    "-----------sample description-----------\n"  # noqa
    "{sample_description}\n"  # noqa
    "-----------labels-----------\n"  # noqa
    "{labels}\n"  # noqa
    "---------------------\n"  # noqa
    "Using the sample description above and your knowledge, return the label that "  # noqa
    "is most relevant to the sample.\n"  # noqa
    "Provide label in the following format: 'ANSWER: <label>' and explain why "  # noqa
    "this label was selected.\n"  # noqa
)

DEFAULT_SCENARIO_REGRESSION_TMPL = (  # noqa
    "The sceario is {scenario} with tabular data. "  # noqa
    "We first provide sample description contains columns and corresponding values, "  # noqa
    "and then we provide context information to define a regression task.\n"  # noqa
    "-----------sample description-----------\n"  # noqa
    "{sample_description}\n"  # noqa
    "-----------context information-----------\n"  # noqa
    "{context_info}\n"  # noqa
    "---------------------\n"  # noqa
    "Using the sample description above and your knowledge, return a value that "  # noqa
    "is most possible to the sample.\n"  # noqa
    "Provide value in the following format: 'ANSWER: <value>' and explain why "  # noqa
    "this value was predicted.\n"  # noqa
)

DEFAULT_SCENARIO_EXPLANATION_TMPL = (  # noqa
    "The sceario is {scenario} with tabular data. "  # noqa
    "We will provide sample description contains columns and corresponding values\n"   # noqa
    "-----------sample description-----------\n"  # noqa
    "{sample_description}\n"  # noqa
    "---------------------\n"  # noqa
    "read the sample description above and use your knowledge"  # noqa
    "to perform interpretative enhancement of the sample."  # noqa
    "To make it more detailed, contents can include background information, highlight potential impacts and so on.\n"  # noqa
    "Provide value in the following format: 'ANSWER: <explanation>'\n"  # noqa
)
