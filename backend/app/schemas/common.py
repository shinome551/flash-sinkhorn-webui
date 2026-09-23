from enum import StrEnum

from pydantic import BaseModel


class ErrorCode(StrEnum):
    """SPEC 4.5 のエラーコード。末尾2つは仕様外の汎用コード。"""

    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    TOO_MANY_PATCHES = "TOO_MANY_PATCHES"
    INVALID_PARAMS = "INVALID_PARAMS"
    SOLVER_FAILED = "SOLVER_FAILED"
    HTTP_ERROR = "HTTP_ERROR"  # 未定義ルート (404) やメソッド違反 (405) など
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 想定外の例外


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    hint: str | None = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail
