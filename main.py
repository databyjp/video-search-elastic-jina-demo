from omnimodal_search.es import get_client, create_index

client = get_client()
print(client.info())

INDEX = "elastic-wizard"
client.indices.delete(index=INDEX, ignore_unavailable=True)
create_index(client, INDEX, dims=1024)
if client.indices.exists(INDEX):
    print(f"Index `{INDEX}` created")
else:
    print(f"Error - index `{INDEX}` not found")
