select
    touchpoint_id,
    account_id,
    contact_id,
    channel,
    touch_ts,
    cast(touch_ts as date) as touch_date,
    source_type
from {{ ref('int_touchpoints') }}
