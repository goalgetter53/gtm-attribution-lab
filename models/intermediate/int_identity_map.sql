-- Anonymous web visitor -> known CRM contact, via identify calls (form fills, email links).
-- Earliest identify wins; sessions before the identify are stitched retroactively.
with matched as (
    select
        i.anonymous_id,
        c.contact_id,
        i.identified_at,
        row_number() over (
            partition by i.anonymous_id order by i.identified_at, c.contact_id
        ) as rn
    from {{ ref('stg_identifies') }} as i
    inner join {{ ref('stg_contacts') }} as c on c.email = i.email
)

select anonymous_id, contact_id, identified_at
from matched
where rn = 1
