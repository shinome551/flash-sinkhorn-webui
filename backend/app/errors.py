from app.schemas.common import ErrorCode


class AppError(Exception):
    """API エラーとして返す例外。FastAPI には依存せず、コア層からも送出できる。

    ``app.main`` の例外ハンドラが ``{"detail": {code, message, hint}}`` に変換する。
    """

    def __init__(
        self, code: ErrorCode, message: str, *, status_code: int, hint: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.hint = hint


def image_not_found(image_id: str) -> AppError:
    return AppError(
        ErrorCode.IMAGE_NOT_FOUND,
        f"Image not found: {image_id}",
        status_code=404,
        hint="The image may have expired; upload it again.",
    )
