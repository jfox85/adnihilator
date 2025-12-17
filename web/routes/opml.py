"""OPML import/export routes."""

import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..auth import verify_admin
from ..dependencies import get_db
from ..models import Podcast


router = APIRouter(prefix="/opml", tags=["opml"])


def parse_opml(content: bytes) -> list[dict]:
    """Parse OPML file and extract podcast feed URLs.

    Args:
        content: Raw OPML file content

    Returns:
        List of dicts with 'url' and 'title' keys

    Raises:
        ValueError: If OPML is malformed
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    feeds = []

    # OPML structure: opml > body > outline (possibly nested)
    body = root.find("body")
    if body is None:
        raise ValueError("OPML missing <body> element")

    def extract_outlines(element):
        """Recursively extract feed outlines."""
        for outline in element.findall("outline"):
            # Check for xmlUrl attribute (indicates a feed)
            xml_url = outline.get("xmlUrl") or outline.get("xmlurl")
            if xml_url:
                feeds.append({
                    "url": xml_url,
                    "title": outline.get("title") or outline.get("text") or "",
                })
            # Recurse into nested outlines (folders)
            extract_outlines(outline)

    extract_outlines(body)
    return feeds


def generate_opml(podcasts: list[Podcast], base_url: str) -> bytes:
    """Generate OPML file with clean feed URLs.

    Args:
        podcasts: List of Podcast objects
        base_url: Base URL for generating feed URLs

    Returns:
        OPML file content as bytes
    """
    opml = ET.Element("opml", version="2.0")

    # Head section
    head = ET.SubElement(opml, "head")
    ET.SubElement(head, "title").text = "AdNihilator - Ad-Free Podcast Feeds"
    ET.SubElement(head, "dateCreated").text = datetime.utcnow().strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )

    # Body section
    body = ET.SubElement(opml, "body")

    for podcast in podcasts:
        feed_url = f"{base_url}/feed/{podcast.feed_token}.xml"
        ET.SubElement(
            body,
            "outline",
            type="rss",
            text=podcast.title or "Untitled",
            title=podcast.title or "Untitled",
            xmlUrl=feed_url,
            htmlUrl=podcast.source_rss_url,
        )

    # Generate XML with declaration
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        opml, encoding="unicode"
    ).encode("utf-8")


@router.post("/import")
async def import_opml(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Import podcasts from an OPML file.

    Parses the uploaded OPML file and adds any new podcast feeds.
    Existing feeds (by URL) are skipped.

    Returns:
        Summary of imported and skipped feeds
    """
    if not file.filename or not file.filename.lower().endswith((".opml", ".xml")):
        raise HTTPException(
            status_code=400,
            detail="File must be an OPML file (.opml or .xml)",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        feeds = parse_opml(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not feeds:
        raise HTTPException(status_code=400, detail="No feeds found in OPML file")

    # Get existing feed URLs
    existing_urls = {p.source_rss_url for p in db.query(Podcast.source_rss_url).all()}

    imported = []
    skipped = []

    for feed in feeds:
        url = feed["url"]
        if url in existing_urls:
            skipped.append({"url": url, "title": feed["title"], "reason": "already exists"})
            continue

        # Add new podcast with title from OPML (will be updated on sync)
        podcast = Podcast(source_rss_url=url, title=feed["title"] or None)
        db.add(podcast)
        db.commit()

        imported.append({
            "url": url,
            "title": feed["title"] or "Untitled",
            "id": podcast.id,
        })

        existing_urls.add(url)

    return {
        "imported": len(imported),
        "skipped": len(skipped),
        "feeds": {
            "imported": imported,
            "skipped": skipped,
        },
    }


@router.get("/export")
async def export_opml(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
):
    """Export all podcast feeds as an OPML file.

    Generates an OPML file with the ad-free feed URLs for all podcasts.
    This file can be imported into podcast apps to subscribe to the clean feeds.
    """
    podcasts = db.query(Podcast).order_by(Podcast.title).all()

    # Get base URL from request
    base_url = str(request.base_url).rstrip("/")

    opml_content = generate_opml(podcasts, base_url)

    return Response(
        content=opml_content,
        media_type="application/xml",
        headers={
            "Content-Disposition": 'attachment; filename="adnihilator-feeds.opml"',
        },
    )
