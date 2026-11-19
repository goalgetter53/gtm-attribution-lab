-- The conversion event each journey is attributed to.
--   converted accounts: the CRM 'won' timestamp
--   others: closed-lost date, else the data cutoff ({{ var('data_end_ts') }})
-- focus_contact_id is the contact a contact-level model would follow: the opportunity's
-- primary contact, or the earliest-created contact for accounts without an opportunity.
with first_contact as (
    select account_id, arg_min(contact_id, created_at) as contact_id
    from {{ ref('int_contact_accounts') }}
    where account_id is not null
    group by account_id
)

select
    f.account_id,
    f.won_at is not null as converted,
    coalesce(f.won_at, o.closed_at, timestamp '{{ var('data_end_ts') }}') as conversion_at,
    f.lead_at,
    f.mql_at,
    f.opportunity_at,
    coalesce(o.primary_contact_id, fc.contact_id) as focus_contact_id,
    o.amount
from {{ ref('fct_funnel_stages') }} as f
left join {{ ref('fct_opportunities') }} as o on o.account_id = f.account_id
left join first_contact as fc on fc.account_id = f.account_id
