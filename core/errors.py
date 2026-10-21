"""Excepciones propias: errores atribuibles al usuario o a los datos."""


class UserError(Exception):
    """Error que se muestra al usuario como aviso (sin traza)."""


class AudioError(UserError):
    """Archivo WAV inválido o no soportado."""


class ParamsError(UserError):
    """Parámetro global fuera de rango o mal escrito."""
