-- materialized: view
-- CRM activities. Unanswered dials never reached the buyer, so they are not touches.
select
    activity_id,
    contact_id,
    activity_type,
    outcome,
    cast(activity_ts as timestamp) as activity_ts,
    case activity_type
        when 'call' then 'sdr_call'
        when 'webinar_attended' then 'webinar'
        when 'event_attended' then 'events'
    end as channel,
    (activity_type = 'call' and outcome = 'connected')
        or activity_type in ('webinar_attended', 'event_attended') as is_touch
from {{ source('raw', 'activities') }}
