from lightning.app.utilities.network import LightningClient

client = LightningClient()

projects = client.projects_service_list_memberships()

counter = 0

while True:

    for project in projects.memberships:

        print(project.name)

        cloudspaces = client.cloud_space_service_list_cloud_spaces(project_id=project.project_id)

        if len(cloudspaces.cloudspaces) > 2:

            for cloud_space in cloudspaces.cloudspaces:
                print(cloud_space.name)
                resp = client.cloud_space_service_delete_cloud_space(project_id=project.project_id, id=cloud_space.id)
                counter += 1
                print(counter, resp)
