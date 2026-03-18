from typing import Any, Dict, Optional


def api_success(data: Any) -> Dict[str, Any]:
	return {"success": True, "data": data}


def api_error(code: str, message: str, details: Optional[Any] = None) -> Dict[str, Any]:
	error: Dict[str, Any] = {"code": code, "message": message}
	if details is not None:
		error["details"] = details
	return {"success": False, "data": None, "error": error}
