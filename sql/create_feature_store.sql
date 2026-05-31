CREATE TABLE feature_store(
    sample_id serial primary key,
    cap_texture varchar(1),
    spore_pattern varchar(1),
    stem_flexibility varchar(1),
    ring_thickness varchar(1),
    cap_shape varchar(1),
    cap_surface varchar(1),
    cap_color varchar(1),
    gill_attachment varchar(1),
    stalk_shape varchar(1),
    veil_type varchar(1),
    veil_color varchar(1),
    ring_number varchar(1),
    population varchar(1),
    habitat varchar(1),
    class varchar(1),
    date_created timestamp default now()
)