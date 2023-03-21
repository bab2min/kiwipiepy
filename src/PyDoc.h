#pragma once

#define DOC_KO

#define DOC_SIGNATURE_EN(name, signature, en) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN(name, en) PyDoc_STRVAR(name, en)
#ifdef DOC_KO
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" ko)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, ko)
#else
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, en)
#endif

DOC_SIGNATURE_EN_KO(Token__doc__,
    "Token(self)",
    u8R""()"",
    u8R""(Token은 분석 결과 얻어진 형태소 정보를 담는 데이터 클래스입니다. (`form`, `tag`, `start`, `len`) 형태의 길이 4의 `tuple`로 변환 가능합니다.)""
);

DOC_VARIABLE_EN_KO(Token_form__doc__,
    u8R""()"",
    u8R""(형태소의 형태)""
);

DOC_VARIABLE_EN_KO(Token_tag__doc__,
    u8R""()"",
    u8R""(형태소의 품사 태그)""
);

DOC_VARIABLE_EN_KO(Token_start__doc__,
    u8R""()"",
    u8R""(형태소의 입력 텍스트 내 시작 위치 (문자 단위))""
);

DOC_VARIABLE_EN_KO(Token_end__doc__,
    u8R""()"",
    u8R""(형태소의 입력 텍스트 내 끝 위치 (문자 단위))""
);

DOC_VARIABLE_EN_KO(Token_len__doc__,
    u8R""()"",
    u8R""(형태소의 입력 텍스트 내 차지 길이 (문자 단위))""
);

DOC_VARIABLE_EN_KO(Token_id__doc__,
    u8R""()"",
    u8R""(형태소의 내부 고유 ID)""
);

DOC_VARIABLE_EN_KO(Token_word_position__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.10.2

형태소의 입력 텍스트 내 어절 위치 (공백 기준, 문장별로 0부터 시작))""
);

DOC_VARIABLE_EN_KO(Token_sent_position__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.10.3

형태소의 입력 텍스트 내 문장 번호 (0부터 시작))""
);

DOC_VARIABLE_EN_KO(Token_sub_sent_position__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.14.0

형태소가 안긴 문장에 속한 경우, 현 문장 내의 안긴 문장 번호. (1부터 시작. 0일 경우 안긴 문장이 아님))""
);

DOC_VARIABLE_EN_KO(Token_line_number__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.10.3

형태소의 입력 텍스트 내 줄 번호 (0부터 시작))""
);

DOC_VARIABLE_EN_KO(Token_base_form__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.11.0

이형태의 경우 원본 형태소의 형태. 일반 형태소의 경우 `form`과 동일.)""
);

DOC_VARIABLE_EN_KO(Token_base_id__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.11.0

이형태의 경우 원본 형태소의 고유 ID. 일반 형태소의 경우 `id`와 동일.)""
);


DOC_VARIABLE_EN_KO(Token_tagged_form__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.11.1

form과 tag를 `형태/품사태그`꼴로 합쳐서 반환합니다.)""
);

DOC_VARIABLE_EN_KO(Token_score__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.12.0

현재 형태소의 언어 모델 상의 점수를 반환합니다.)""
);

DOC_VARIABLE_EN_KO(Token_regularity__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.12.0

동/형용사가 규칙 활용하는 경우 True, 불규칙 활용하는 경우 False를 반환합니다.
그 외의 품사에 대해서는 None을 반환합니다.)""
);

DOC_VARIABLE_EN_KO(Token_lemma__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.15.0

동/형용사의 경우 '-다'를 붙여서 형태소의 사전 표제형을 반환합니다.)""
);

DOC_VARIABLE_EN_KO(Token_typo_cost__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.13.0

현재 형태소의 오타 교정 비용을 반환합니다.)""
);


DOC_VARIABLE_EN_KO(Token_raw_form__doc__,
    u8R""()"",
    u8R""(.. versionadded:: 0.13.0

텍스트 상에 실제로 등장한 형태. 오타가 교정된 경우 `form`과 다를 수 있음.)""
);
