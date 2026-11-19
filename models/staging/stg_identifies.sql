-- materialized: view
select
    anonymous_id,
    lower(trim(email)) as email,
    cast(identified_at as timestamp) as identified_at
from {{ source('raw', 'identifies') }}
