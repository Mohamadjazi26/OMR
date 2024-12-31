import enum
import typing as tp
# from hazm import *
import alphabet
from geometry_utils import Orientation

KEY_STUDENT_ID = "11"
GRID_HORIZONTAL_CELLS = 39 #(36,48)
GRID_VERTICAL_CELLS = 52


class Field(enum.Enum):
    """Fields that exist on the bubble sheet."""
    LAST_NAME = enum.auto()
    FIRST_NAME = enum.auto()
    MIDDLE_NAME = enum.auto()
    TEST_FORM_CODE = enum.auto()
    STUDENT_ID = enum.auto()
    COURSE_ID = enum.auto()
    IMAGE_FILE = enum.auto()


class VirtualField(enum.Enum):
    SCORE = enum.auto()
    POINTS = enum.auto()


RealOrVirtualField = tp.Union[Field, VirtualField]


class FieldType(enum.Enum):
    LETTER = enum.auto()
    NUMBER = enum.auto()


class GridGroupInfo:

    horizontal_start: int
    vertical_start: int
    num_fields: int
    field_length: int
    fields_type: FieldType
    field_orientation: Orientation

    def __init__(self,
                 horizontal_start: int,
                 vertical_start: int,
                 num_fields: int = 1,
                 fields_type: FieldType = FieldType.NUMBER,
                 field_length: tp.Optional[int] = None,
                 field_orientation: Orientation = Orientation.VERTICAL):
        self.horizontal_start = horizontal_start
        self.vertical_start = vertical_start
        self.num_fields = num_fields
        if field_length is not None:
            self.field_length = field_length
        elif fields_type is FieldType.LETTER:
            self.field_length = alphabet.LENGTH
        else:
            self.field_length = 10
        self.fields_type = fields_type
        self.field_orientation = field_orientation


class FormVariant:
    fields: tp.Dict[Field, tp.Optional[GridGroupInfo]]
    questions: tp.List[GridGroupInfo]
    questions_per_column: int
    num_questions: int

    def __init__(self, fields: tp.Dict[Field, tp.Optional[GridGroupInfo]],
                 questions: tp.List[GridGroupInfo],
                 questions_per_column: int = 25):
        self.fields = fields
        self.questions = questions
        self.questions_per_column = questions_per_column
        self.num_questions = len(questions)


form_75q = FormVariant(
    {
        Field.LAST_NAME:
        GridGroupInfo(1, 3, 12, fields_type=FieldType.LETTER),
        Field.FIRST_NAME:
        GridGroupInfo(14, 3, 6, fields_type=FieldType.LETTER),
        Field.MIDDLE_NAME:
        GridGroupInfo(21, 3, 2, fields_type=FieldType.LETTER),
        Field.STUDENT_ID:
        GridGroupInfo(28, 9, 6),
        Field.COURSE_ID:
        GridGroupInfo(25, 16, 10),

        #-------------------------------------------------------------------------------

        Field.TEST_FORM_CODE:
        GridGroupInfo(27,
                      28,
                      fields_type=FieldType.NUMBER,
                      field_length=6,
                      field_orientation=Orientation.HORIZONTAL)
    }, [
        GridGroupInfo(3 + (6 * (i // 25)),
                      21 + i % 25,
                      fields_type=FieldType.NUMBER,
                      field_length=4,
                      field_orientation=Orientation.HORIZONTAL)
        for i in range(75)
    ])

form_150q = FormVariant(
    {
        Field.STUDENT_ID:
        GridGroupInfo(25, 3, 10),
        Field.COURSE_ID:
        GridGroupInfo(14, 3, 10),
        Field.TEST_FORM_CODE:
        GridGroupInfo(4,
                      12,
                      fields_type=FieldType.NUMBER,
                      field_length=6,
                      field_orientation=Orientation.HORIZONTAL)
    }, [
        GridGroupInfo(3 + (6 * (i // 25)),
                      21 + i - (25 * (i // 25)),
                      fields_type=FieldType.NUMBER,
                      field_length=4,
                      field_orientation=Orientation.HORIZONTAL)
        for i in range(150)
    ],
    questions_per_column=25
)


class dimensions:
    vertical_cells = GRID_VERTICAL_CELLS
    horizontal_cells = GRID_HORIZONTAL_CELLS
