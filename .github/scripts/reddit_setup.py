"""Saca el refresh token de Reddit que necesita announce.py. Se ejecuta UNA vez.

Autorizas la app desde el navegador y Reddit devuelve un refresh token PERMANENTE
que solo sirve para publicar (scope `submit`). Asi tu contraseña de Reddit no se
guarda en ningun sitio, y funciona aunque tengas 2FA.

    REDDIT_CLIENT_ID=xxx python reddit_setup.py

El client secret se pide por teclado (oculto). Los tokens no se imprimen: van
directos a los secrets del repo con `gh secret set`.

Requisito previo: crea la app en https://www.reddit.com/prefs/apps
  - tipo: **web app**  (NO "script": el script solo permite password grant)
  - redirect uri: http://localhost:8765/callback
"""

from __future__ import annotations

import base64
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
SCOPES = "submit"  # lo minimo: publicar. Nada de leer ni de tocar la cuenta.
REPO = "LeandroPG19/cuba-memorys"
USER_AGENT = "linux:cuba-memorys-announce:1.0 (by /u/{user})"

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
            else "Reddit no ha dado ningun codigo."
        )
        self.wfile.write(mensaje.encode("utf-8"))

    def log_message(self, *_args: object) -> None:
        pass


def set_secret(name: str, value: str) -> None:
    subprocess.run(
        ["gh", "secret", "set", name, "-R", REPO],
        input=value,
        text=True,
        check=True,
        capture_output=True,
    )
    print(f"  {name}: guardado")


def main() -> int:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    username = os.environ.get("REDDIT_USERNAME")
    if not client_id or not username:
        print("hacen falta REDDIT_CLIENT_ID y REDDIT_USERNAME", file=sys.stderr)
        return 1

    client_secret = os.environ.get("REDDIT_CLIENT_SECRET") or getpass.getpass(
        "Pega el secret de la app de Reddit (no se vera al escribir): "
    )
    if not client_secret:
        print("sin secret no puedo seguir", file=sys.stderr)
        return 1

    user_agent = USER_AGENT.format(user=username)

    state = secrets.token_urlsafe(16)
    server = http.server.HTTPServer(("localhost", 8765), CallbackHandler)
    server.expected_state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.handle_request, daemon=True).start()

    auth_url = "https://www.reddit.com/api/v1/authorize?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "state": state,
            "redirect_uri": REDIRECT_URI,
            "duration": "permanent",  # <- lo que hace que el refresh token no caduque
            "scope": SCOPES,
        }
    )
    print("Abriendo el navegador para autorizar la app...")
    print(f"Si no se abre solo, entra aqui:\n{auth_url}\n")
    webbrowser.open(auth_url)

    if not _done.wait(timeout=300):
        print("nadie autorizo la app en 5 minutos: lo dejo", file=sys.stderr)
        return 1
    if _code is None:
        print(
            "Reddit no devolvio ningun codigo (¿le diste a Decline?)", file=sys.stderr
        )
        return 1
    if not _state_ok:
        print("el `state` no coincide: aborto por seguridad", file=sys.stderr)
        return 1

    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            "https://www.reddit.com/api/v1/access_token",
            headers={"Authorization": f"Basic {basic}", "User-Agent": user_agent},
            data={
                "grant_type": "authorization_code",
                "code": _code,
                "redirect_uri": REDIRECT_URI,
            },
        )
        if response.status_code != 200:
            print(
                f"Reddit no dio el token: {response.status_code} {response.text}",
                file=sys.stderr,
            )
            return 1
        tokens = response.json()

    refresh = tokens.get("refresh_token")
    if not refresh:
        print(
            "Reddit no ha dado refresh token. ¿Creaste la app como 'web app' "
            "y autorizaste con duration=permanent?",
            file=sys.stderr,
        )
        return 1

    print(f"\nGuardando los secrets en {REPO} (ninguno se imprime):\n")
    try:
        set_secret("REDDIT_CLIENT_ID", client_id)
        set_secret("REDDIT_CLIENT_SECRET", client_secret)
        set_secret("REDDIT_REFRESH_TOKEN", refresh)
        set_secret("REDDIT_USERNAME", username)
    except (subprocess.CalledProcessError, FileNotFoundError) as why:
        print(f"\nno pude guardarlos con gh: {why}", file=sys.stderr)
        return 1

    print(
        f"\nHecho. El refresh token no caduca y solo puede publicar (scope: {SCOPES})."
    )
    print("Tu contraseña de Reddit no se ha guardado en ningun sitio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
