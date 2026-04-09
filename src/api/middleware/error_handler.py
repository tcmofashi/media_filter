"""Error handling middleware for unified API error responses."""

import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.logger import get_logger
from src.models.frozen_clip_engine import FrozenClipEngineError, ModelNotLoadedError

logger = get_logger(__name__)


class ErrorCode:
    """Standard error codes for the API."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    VALUE_ERROR = "VALUE_ERROR"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def build_error_response(
    code: str,
    message: str,
    details: Any | None = None,
) -> dict[str, Any]:
    """Build a standardized error response payload."""
    error_dict: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if details is not None:
        error_dict["details"] = details
    return {"error": error_dict}


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle request validation errors."""
    if not isinstance(exc, ValidationError):
        raise exc
    details = exc.errors()
    logger.warning("Validation error on %s: %s", request.url.path, details)
    return JSONResponse(
        status_code=400,
        content=build_error_response(
            code=ErrorCode.VALIDATION_ERROR,
            message="Request validation failed",
            details=details,
        ),
    )


async def value_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle ValueError exceptions."""
    if not isinstance(exc, ValueError):
        raise exc
    logger.warning("Value error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=build_error_response(
            code=ErrorCode.VALUE_ERROR,
            message=str(exc),
        ),
    )


async def model_not_loaded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle model-not-loaded errors."""
    if not isinstance(exc, ModelNotLoadedError):
        raise exc
    logger.warning("Model not loaded on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content=build_error_response(
            code=ErrorCode.MODEL_NOT_LOADED,
            message=str(exc),
        ),
    )


async def inference_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Frozen CLIP inference errors."""
    if not isinstance(exc, FrozenClipEngineError):
        raise exc
    logger.error("Inference error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=build_error_response(
            code=ErrorCode.INFERENCE_ERROR,
            message=str(exc),
        ),
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle HTTPException with standardized format."""
    if not isinstance(exc, HTTPException):
        raise exc
    logger.warning("HTTP %s on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_response(
            code=str(exc.status_code),
            message=str(exc.detail),
        ),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other unexpected exceptions."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "Unhandled exception on %s%s: %s\n%s",
        request.url.path,
        f" (request_id: {request_id})" if request_id else "",
        exc,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content=build_error_response(
            code=ErrorCode.INTERNAL_ERROR,
            message="An internal error occurred. Please try again later.",
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(ValidationError, validation_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
    app.add_exception_handler(FrozenClipEngineError, inference_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Error handlers registered")
