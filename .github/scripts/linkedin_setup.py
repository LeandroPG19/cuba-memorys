"""Saca las credenciales de LinkedIn que necesita announce.py. Se ejecuta UNA vez.

Hace el baile de OAuth en local: abre el navegador, recoge el codigo, lo cambia
por los tokens y consulta tu URN de persona. Los tokens NO se imprimen: van
directos a los secrets del repo con `gh secret set`, asi que no acaban ni en la
pantalla ni en el historial del shell.

    LINKEDIN_CLIENT_ID=772im4q6sicwuu python linkedin_setup.py

El client secret se pide por teclado (oculto). Requisito previo: en la app de
LinkedIn Developers, en Auth > Redirect URLs, tiene que estar dada de alta
exactamente esta URL:

    http://localhost:8765/callback
"""

from __future__ import annotations

import getpass
import http.server
import os
import secrets
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

import httpx

REDIRECT_URI = "http://localhost:8765/callback"
SCOPES = "openid profile w_member_social"
REPO = "LeandroPG19/cuba-memorys"

_code: str | None = None
_state_ok = False
_done = threading.Event()


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802  (lo impone la stdlib)
        global _code, _state_ok

        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _code = query.get("code", [None])[0]
        _state_ok = query.get("state", [None])[0] == self.server.expected_state  # type: ignore[attr-defined]
        _done.set()

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        mensaje = (
            "Listo, ya puedes cerrar esta pestaña."
            if _code
            else "No ha llegado ningun codigo."
        )
        self.wfile.write(mensaje.encode("utf-8"))

    def log_message(self, *_args: object) -> None:
        pass  # sin ruido en la consola


def set_secret(name: str, value: str) -> None:
    """Guarda el secret en el repo sin que pase por la pantalla."""
    subprocess.run(
        ["gh", "secret", "set", name, "-R", REPO],
        input=value,
        text=True,
        check=True,
        capture_output=True,
    )
    print(f"  {name}: guardado")


def main() -> int:
    client_id = os.environ.get("LINKEDIN_CLIENT_ID")
    if not client_id:
        print("falta LINKEDIN_CLIENT_ID", file=sys.stderr)
        return 1

    client_secret = os.environ.get("LINKEDIN_CLIENT_SECRET") or getpass.getpass(
        "Pega el Primary Client Secret de la app (no se vera al escribir): "
    )
    if not client_secret:
        print("sin client secret no puedo seguir", file=sys.stderr)
        return 1

    state = secrets.token_urlsafe(16)
    server = http.server.HTTPServer(("localhost", 8765), CallbackHandler)
    server.expected_state = state  # type: ignore[attr-defined]

    def _serve_until_callback() -> None:
        # Una conexion espuria (probe, health-check, request malformada) no
        # invoca do_GET y por tanto no marca _done: seguimos escuchando en
        # vez de agotar el listener one-shot antes de que llegue el redirect
        # real de LinkedIn.
        while not _done.is_set():
            server.handle_request()

    threading.Thread(target=_serve_until_callback, daemon=True).start()

    auth_url = (
        "https://www.linkedin.com/oauth/v2/authorization?"
        + urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": REDIRECT_URI,
                "state": state,
                "scope": SCOPES,
            }
        )
    )
    print("Abriendo el navegador para autorizar la app...")
    print(f"Si no se abre solo, entra aqui:\n{auth_url}\n")
    webbrowser.open(auth_url)

    if not _done.wait(timeout=300):
        print("nadie autorizo la app en 5 minutos: lo dejo", file=sys.stderr)
        return 1

    if _code is None:
        print("LinkedIn no devolvio ningun codigo", file=sys.stderr)
        return 1

    if not _state_ok:
        print("el `state` no coincide: aborto por seguridad", file=sys.stderr)
        return 1

    with httpx.Client(timeout=30.0) as client:
        token_response = client.post(
            "https://www.linkedin.com/oauth/v2/accessToken",
            data={
                "grant_type": "authorization_code",
                "code": _code,
                "redirect_uri": REDIRECT_URI,
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        if token_response.status_code != 200:
            print(f"LinkedIn no dio el token: {token_response.text}", file=sys.stderr)
            return 1
        tokens = token_response.json()

        userinfo = client.get(
            "https://api.linkedin.com/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        if userinfo.status_code != 200:
            print(f"no pude leer tu perfil: {userinfo.text}", file=sys.stderr)
            return 1
        person_urn = f"urn:li:person:{userinfo.json()['sub']}"

    print(f"\nGuardando los secrets en {REPO} (ninguno se imprime):\n")
    guardar = {
        "LINKEDIN_CLIENT_ID": client_id,
        "LINKEDIN_CLIENT_SECRET": client_secret,
        "LINKEDIN_PERSON_URN": person_urn,
        "LINKEDIN_ACCESS_TOKEN": tokens["access_token"],
    }
    refresh = tokens.get("refresh_token")
    if refresh:
        guardar["LINKEDIN_REFRESH_TOKEN"] = refresh

    try:
        for name, value in guardar.items():
            set_secret(name, value)
    except (subprocess.CalledProcessError, FileNotFoundError) as why:
        print(f"\nno pude guardarlos con gh: {why}", file=sys.stderr)
        print("¿estas autenticado? prueba `gh auth status`", file=sys.stderr)
        return 1

    if refresh:
        print("\nHay refresh token: announce.py renovara el acceso solo.")
    else:
        expires_days = tokens.get("expires_in", 0) // 86400
        print(
            f"\nOJO: tu app no da refresh token y el access token caduca en {expires_days} dias."
            "\nCuando expire, vuelve a ejecutar este script."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
