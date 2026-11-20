"""Small Windows Credential Manager wrapper for local API credentials."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

_TARGET = "StreamClipper:youtube_api_key"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def available() -> bool:
    return os.name == "nt"


def _api():
    if not available():
        raise OSError("Windows Credential Manager is available only on Windows.")
    return ctypes.WinDLL("Advapi32.dll", use_last_error=True)


def get_secret() -> str | None:
    api = _api()
    api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                              ctypes.POINTER(ctypes.POINTER(_Credential))]
    api.CredReadW.restype = wintypes.BOOL
    api.CredFree.argtypes = [ctypes.c_void_p]
    cred = ctypes.POINTER(_Credential)()
    if not api.CredReadW(_TARGET, _CRED_TYPE_GENERIC, 0, ctypes.byref(cred)):
        error = ctypes.get_last_error()
        if error == 1168:  # ERROR_NOT_FOUND
            return None
        raise ctypes.WinError(error)
    try:
        raw = ctypes.string_at(cred.contents.CredentialBlob, cred.contents.CredentialBlobSize)
        return raw.decode("utf-8")
    finally:
        api.CredFree(cred)


def set_secret(value: str) -> None:
    api = _api()
    api.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
    api.CredWriteW.restype = wintypes.BOOL
    blob = value.encode("utf-8")
    buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
    cred = _Credential()
    cred.Type = _CRED_TYPE_GENERIC
    cred.TargetName = _TARGET
    cred.CredentialBlobSize = len(blob)
    cred.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    cred.Persist = _CRED_PERSIST_LOCAL_MACHINE
    cred.UserName = "StreamClipper"
    if not api.CredWriteW(ctypes.byref(cred), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def delete_secret() -> None:
    api = _api()
    api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    api.CredDeleteW.restype = wintypes.BOOL
    if not api.CredDeleteW(_TARGET, _CRED_TYPE_GENERIC, 0):
        error = ctypes.get_last_error()
        if error != 1168:
            raise ctypes.WinError(error)
