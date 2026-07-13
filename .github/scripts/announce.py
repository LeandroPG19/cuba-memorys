"""Anuncia una release de cuba-memorys en LinkedIn y Reddit.

El texto NO se inventa: sale de la seccion del CHANGELOG.md correspondiente a la
version que se publica. El lede (el parrafo que va justo debajo del titulo de la
version) es el gancho, y los titulares en negrita de cada punto son el cuerpo.

Uso:
    python announce.py --version 0.11.1 --dry-run
    python announce.py --version 0.11.1            # publica de verdad

Variables de entorno (las que falten, desactivan esa plataforma):
    LINKEDIN_ACCESS_TOKEN   token de acceso (o LINKEDIN_REFRESH_TOKEN + CLIENT_ID/SECRET)
    LINKEDIN_PERSON_URN     urn:li:person:XXXX
    REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USERNAME / REDDIT_PASSWORD
    REDDIT_SUBREDDITS       lista separada por comas, p.ej. "rust,opensource"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_URL = "https://github.com/LeandroPG19/cuba-memorys"
LINKEDIN_VERSION = "202506"  # cabecera LinkedIn-Version (YYYYMM)
USER_AGENT = "linux:cuba-memorys-announce:1.0 (by /u/{user})"

# LinkedIn "Little Text": estos caracteres hay que escaparlos en `commentary`.
LINKEDIN_RESERVED = r"\(){}[]<>@|#*_~"


class AnnounceError(RuntimeError):
    """Fallo al componer o publicar el anuncio."""


# ─────────────────────────── el texto sale del CHANGELOG ───────────────────────────


@dataclass(frozen=True)
class Release:
    version: str
    lede: str
    headlines: list[str]

    @property
    def url(self) -> str:
        return f"{REPO_URL}/releases/tag/v{self.version}"


def parse_changelog(changelog: str, version: str) -> Release:
    """Extrae el lede y los titulares de la seccion `## [version]`."""
    lines = changelog.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"## [{version}]")),
        None,
    )
    if start is None:
        raise AnnounceError(f"CHANGELOG.md no tiene una seccion `## [{version}]`")

    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")),
        len(lines),
    )
    section = lines[start + 1 : end]

    # El lede: los parrafos entre el titulo de la version y el primer `### `.
    lede_lines: list[str] = []
    for line in section:
        if line.startswith("### "):
            break
        lede_lines.append(line)
    lede = " ".join(" ".join(lede_lines).split())

    # Los titulares: el primer **negrita** de cada punto de la lista.
    headlines = [
        match.group(1).strip()
        for line in section
        if line.lstrip().startswith("- ")
        and (match := re.search(r"\*\*(.+?)\*\*", line))
    ]

    if not lede and not headlines:
        raise AnnounceError(
            f"la seccion `## [{version}]` del CHANGELOG no tiene ni lede ni puntos: "
            "no hay nada que publicar"
        )

    return Release(
        version=version,
        lede=strip_markdown(lede),
        headlines=[strip_markdown(h) for h in headlines],
    )


def strip_markdown(text: str) -> str:
    """Quita el marcado que en un post se veria como ruido."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [texto](url) -> texto
    text = re.sub(r"[*_`]+", "", text)  # negritas, cursivas, code
    return text.strip()


# ─────────────────────────── composicion por plataforma ───────────────────────────


def linkedin_text(release: Release) -> str:
    parts = [f"cuba-memorys v{release.version}"]
    if release.lede:
        parts.append(release.lede)
    if release.headlines:
        parts.append("\n".join(f"— {h}" for h in release.headlines))
    parts.append(release.url)
    return "\n\n".join(parts)


def reddit_title(release: Release) -> str:
    title = f"cuba-memorys v{release.version}"
    if release.lede:
        title = f"{title} — {release.lede}"
    return title[:300]  # Reddit corta en 300


def reddit_body(release: Release) -> str:
    parts = []
    if release.headlines:
        parts.append("\n".join(f"- {h}" for h in release.headlines))
    parts.append(f"[Release notes]({release.url})")
    return "\n\n".join(parts)


# ─────────────────────────────────── LinkedIn ───────────────────────────────────


def escape_little_text(text: str) -> str:
    return "".join(f"\\{c}" if c in LINKEDIN_RESERVED else c for c in text)


def linkedin_token(client: httpx.Client) -> str:
    """Token de acceso: renovado con el refresh token si lo hay, o el fijo."""
    refresh = os.environ.get("LINKEDIN_REFRESH_TOKEN")
    if not refresh:
        token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
        if not token:
            raise AnnounceError(
                "falta LINKEDIN_ACCESS_TOKEN (o LINKEDIN_REFRESH_TOKEN)"
            )
        return token

    response = client.post(
        "https://www.linkedin.com/oauth/v2/accessToken",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": os.environ["LINKEDIN_CLIENT_ID"],
            "client_secret": os.environ["LINKEDIN_CLIENT_SECRET"],
        },
    )
    if response.status_code != 200:
        raise AnnounceError(
            f"LinkedIn no renovo el token: {response.status_code} {response.text}"
        )
    return response.json()["access_token"]


def post_to_linkedin(client: httpx.Client, release: Release) -> str:
    urn = os.environ.get("LINKEDIN_PERSON_URN")
    if not urn:
        raise AnnounceError("falta LINKEDIN_PERSON_URN")

    response = client.post(
        "https://api.linkedin.com/rest/posts",
        headers={
            "Authorization": f"Bearer {linkedin_token(client)}",
            "LinkedIn-Version": LINKEDIN_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        },
        json={
            "author": urn,
            "commentary": escape_little_text(linkedin_text(release)),
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        },
    )
    if response.status_code != 201:
        raise AnnounceError(
            f"LinkedIn rechazo el post: {response.status_code} {response.text}"
        )
    return response.headers.get("x-restli-id", "publicado")


# ──────────────────────────────────── Reddit ────────────────────────────────────


def reddit_token(client: httpx.Client, user_agent: str) -> str:
    response = client.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(os.environ["REDDIT_CLIENT_ID"], os.environ["REDDIT_CLIENT_SECRET"]),
        data={
            "grant_type": "password",
            "username": os.environ["REDDIT_USERNAME"],
            "password": os.environ["REDDIT_PASSWORD"],
        },
        headers={"User-Agent": user_agent},
    )
    if response.status_code != 200:
        raise AnnounceError(
            f"Reddit no dio token: {response.status_code} {response.text}"
        )
    return response.json()["access_token"]


def post_to_reddit(
    client: httpx.Client, release: Release, subreddits: list[str]
) -> list[str]:
    user_agent = USER_AGENT.format(user=os.environ["REDDIT_USERNAME"])
    headers = {
        "Authorization": f"bearer {reddit_token(client, user_agent)}",
        "User-Agent": user_agent,
    }

    posted = []
    for i, subreddit in enumerate(subreddits):
        if i:
            time.sleep(10)  # no dispares seguido: los mods lo leen como spam

        response = client.post(
            "https://oauth.reddit.com/api/submit",
            headers=headers,
            data={
                "sr": subreddit,
                "kind": "self",
                "title": reddit_title(release),
                "text": reddit_body(release),
                "api_type": "json",
            },
        )
        if response.status_code != 200:
            raise AnnounceError(
                f"r/{subreddit}: {response.status_code} {response.text}"
            )

        payload = response.json().get("json", {})
        if errors := payload.get("errors"):
            raise AnnounceError(f"r/{subreddit} rechazo el post: {errors}")
        posted.append(payload.get("data", {}).get("url", f"r/{subreddit}"))

    return posted


# ───────────────────────────────────── main ─────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True, help="version a anunciar, p.ej. 0.11.1"
    )
    parser.add_argument("--changelog", default="CHANGELOG.md", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="ensena los textos y no publica"
    )
    args = parser.parse_args()

    version = args.version.removeprefix("v")

    try:
        release = parse_changelog(args.changelog.read_text(encoding="utf-8"), version)
    except (OSError, AnnounceError) as why:
        print(f"error: {why}", file=sys.stderr)
        return 1

    print("── LinkedIn ──")
    print(linkedin_text(release))
    print("\n── Reddit ──")
    print(reddit_title(release))
    print()
    print(reddit_body(release))

    if args.dry_run:
        print("\n(dry-run: no se ha publicado nada)")
        return 0

    subreddits = [
        s.strip()
        for s in os.environ.get("REDDIT_SUBREDDITS", "").split(",")
        if s.strip()
    ]
    failures: list[str] = []

    with httpx.Client(timeout=30.0) as client:
        if os.environ.get("LINKEDIN_PERSON_URN"):
            try:
                print(f"\nLinkedIn: {post_to_linkedin(client, release)}")
            except (AnnounceError, httpx.HTTPError, KeyError) as why:
                failures.append(f"LinkedIn: {why}")
        else:
            print("\nLinkedIn: sin credenciales, me lo salto")

        if subreddits and os.environ.get("REDDIT_CLIENT_ID"):
            try:
                for url in post_to_reddit(client, release, subreddits):
                    print(f"Reddit: {url}")
            except (AnnounceError, httpx.HTTPError, KeyError) as why:
                failures.append(f"Reddit: {why}")
        else:
            print("Reddit: sin credenciales o sin subreddits, me lo salto")

    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
