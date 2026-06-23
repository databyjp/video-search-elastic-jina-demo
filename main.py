from omnimodal_search.es import get_client, create_index

# Connect to elastic
client = get_client()
print(client.info())

# Create index
INDEX = "elastic-wizard"
if client.indices.exists(INDEX):
    print(f"Index `{INDEX}` exists, skipping creation")
else:
    create_index(client, INDEX, dims=1024)
    assert client.indices.exists(INDEX)
    print(f"Index `{INDEX}` created")

# Embed objects
