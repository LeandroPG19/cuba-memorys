"""Saca las credenciales de LinkedIn que necesita announce.py. Se ejecuta UNA vez.

Hace el baile de OAuth en local: abre el navegador, recoge el codigo, lo cambia
por los tokens y consulta tu URN de persona. Al final imprime exactamente lo que
hay que pegar en los secrets del repo.

    LINKEDIN_CLIENT_ID=xxx LINKEDIN_CLIENT_SECRET=yyy python linkedin_setup.py

Requisito previo: en la app de LinkedIn Developers, en Auth > Redirect URLs,
tiene que estar dada de alta exactamente esta URL:

    http://localhost:8765/callback
"""

from __future__ import annotations

import http.server
import os
import secrets
import sys
import threading
import urllib.parse
import webbrowser

import httpx

REDIRECT_URI = "http://localhost:8765/callback"
SCOPES = "openid profile w_member_social"

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


def main() -> int:
    client_id = os.environ.get("LINKEDIN_CLIENT_ID")
    client_secret = os.environ.get("LINKEDIN_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("faltan LINKEDIN_CLIENT_ID y LINKEDIN_CLIENT_SECRET", file=sys.stderr)
        return 1

    state = secrets.token_urlsafe(16)
    server = http.server.HTTPServer(("localhost", 8765), CallbackHandler)
    server.expected_state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.handle_request, daemon=True).start()

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

    print("\nPega esto en Settings > Secrets and variables > Actions:\n")
    print(f"  LINKEDIN_CLIENT_ID      {client_id}")
    print(f"  LINKEDIN_CLIENT_SECRET  {client_secret}")
    print(f"  LINKEDIN_PERSON_URN     {person_urn}")
    print(f"  LINKEDIN_ACCESS_TOKEN   {tokens['access_token']}")
    if refresh := tokens.get("refresh_token"):
        print(f"  LINKEDIN_REFRESH_TOKEN  {refresh}")
        print("\nCon el refresh token, announce.py renueva el acceso solo.")
    else:
        expires_days = tokens.get("expires_in", 0) // 86400
        print(
            f"\nOJO: tu app no da refresh token, y el access token caduca en {expires_days} dias."
            "\nTendras que volver a ejecutar este script cuando expire."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
