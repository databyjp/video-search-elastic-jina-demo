"""Delete indexed documents for specific videos by video_id."""

from omnimodal_search.es import get_client

INDEX = "elastic-wizard"

VIDEOS_TO_DELETE = [
    "GSFC_20180703_STEM_m12985_Fenway~large",
    "GSFC_20190419_Earth_m13188_Orbit~large",
    "How Do We Know the Earth Isn't Flat~large",
    "KSC_69-71212-sRGB~medium",
    "NHQ_2019_0301_Celebrating 50 Years of Apollo 9 - A Hell of a Ride~large",
]

client = get_client()

total_deleted = 0
for video_id in VIDEOS_TO_DELETE:
    # Check how many documents match before deleting
    count_resp = client.count(
        index=INDEX,
        query={"term": {"video_id": video_id}},
    )
    doc_count = count_resp.get("count", 0)

    if doc_count == 0:
        print(f"  ⚠️  {video_id}: no documents found")
        continue

    resp = client.delete_by_query(
        index=INDEX,
        query={"term": {"video_id": video_id}},
        refresh=True,
    )

    deleted = resp.get("deleted", 0)
    total_deleted += deleted
    print(f"  ✅ {video_id}: deleted {deleted} document(s)")

print(f"\nTotal documents deleted: {total_deleted}")
