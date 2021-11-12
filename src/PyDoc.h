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

형태소의 입력 텍스트 내 어절 위치 (공백 기준))""
);