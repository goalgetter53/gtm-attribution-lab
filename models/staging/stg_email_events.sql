-- materialized: view
-- Sequencer events. Only clicks and replies are buyer engagement; sends/opens are not touches
-- (opens are inflated by mail-privacy pre-fetching).
select
    event_id,
    lower(trim(email)) as email,
    event_type,
    cast(event_ts as timestamp) as event_ts,
    campaign,
    event_type in ('click', 'reply') as is_touch
from {{ source('raw', 'email_events') }}
