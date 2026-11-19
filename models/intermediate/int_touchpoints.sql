-- Every buyer touch from every source, stitched to a contact and then to an account.
-- Touches whose identity cannot be resolved are dropped here (see mart_identity_coverage).
with web as (
    select
        'web:' || s.session_id as touchpoint_id,
        m.contact_id,
        s.landing_channel as channel,
        s.session_start as touch_ts,
        'web_session' as source_type
    from {{ ref('int_sessions') }} as s
    inner join {{ ref('int_identity_map') }} as m on m.anonymous_id = s.anonymous_id
),

email as (
    select
        'email:' || e.event_id as touchpoint_id,
        c.contact_id,
        'outbound_email' as channel,
        e.event_ts as touch_ts,
        'email_engagement' as source_type
    from {{ ref('stg_email_events') }} as e
    inner join {{ ref('stg_contacts') }} as c on c.email = e.email
    where e.is_touch
),

activity as (
    select
        'activity:' || activity_id as touchpoint_id,
        contact_id,
        channel,
        activity_ts as touch_ts,
        'crm_activity' as source_type
    from {{ ref('stg_activities') }}
    where is_touch
),

unioned as (
    select * from web
    union all
    select * from email
    union all
    select * from activity
)

select
    u.touchpoint_id,
    ca.account_id,
    u.contact_id,
    u.channel,
    u.touch_ts,
    u.source_type
from unioned as u
inner join {{ ref('int_contact_accounts') }} as ca on ca.contact_id = u.contact_id
where ca.account_id is not null
