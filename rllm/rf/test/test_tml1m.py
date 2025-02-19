"""
# Load the data
load from csv files
user_table = Table(...)
movie_table = Table(...)
rating_table = Table(...)

# Construct RelationFrame
relation_frame = RelationFrame(
    tables=[user_table, movie_table, rating_table]
    relations = ...
)

# Dataloader
loader = EntryLoader(relation_frame, batch_size=32, shuffle=True)

# Trainer
for batch in loader:
    out = model(batch)
    ...
"""