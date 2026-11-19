-- Sessionise pageviews: a new session starts after {{ var('session_timeout_minutes') }} idle minutes.
-- The session's channel is the channel of its landing pageview.
with ordered as (
    select
        *,
        lag(event_ts) over (partition by anonymous_id order by event_ts, event_id) as prev_ts
    from {{ ref('stg_web_events') }}
),

flagged as (
    select
        *,
        case
            when prev_ts is null
                or epoch(event_ts) - epoch(prev_ts) > {{ var('session_timeout_minutes') }} * 60
                then 1
            else 0
        end as is_new_session
    from ordered
),

numbered as (
    select
        *,
        sum(is_new_session) over (
            partition by anonymous_id order by event_ts, event_id rows unbounded preceding
        ) as session_number
    from flagged
)

select
    anonymous_id || '-' || cast(session_number as varchar) as session_id,
    anonymous_id,
    session_number,
    min(event_ts) as session_start,
    max(event_ts) as session_end,
    count(*) as pageviews,
    case
        when arg_min(channel, event_ts) = 'internal' then 'direct'
        else arg_min(channel, event_ts)
    end as landing_channel
from numbered
group by anonymous_id, session_number
